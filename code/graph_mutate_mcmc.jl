using Statistics
using Distributions
using LinearAlgebra

include("./graph_decomposition.jl")
include("./normalising_constant_mc.jl")
include("./laplace.jl")

#################################################
#################################################
# mutates the Nxpxp graphs G, with parents Gp,
# for current temperature phi.
##G: Nxpxp, current graphs before mutation
##Gp: Nxpxp, the parents of current graphs
##phi: current temperature
##hyper: hyper-parameters
##parPF: particle filter parameters
##logl_GG: the log of the transition density p(G|Gp)
##logl_GG: the log of the transition density p(Yc|G)
##Yc: the current data
##is_GP: 1 means there is Gp, 0 means no Gp and Gp = zeros(N,p,p)
##pair: is the object [[1,2],[1,3],...,[p-1,p]]

#7.09.2021: now G is modified in-place without being returned
function mutate(Gc, G, Gp, phi, hyper, parPF, logl_GG, logl_GY, logl_GGc, logl_GYc, logl_GGpr, logl_GYpr, Yc, is_Gp, pair, log_acc, decide)

  p = hyper.p; nM = parPF.nM; nE = parPF.nE

  N = size(G)[1]    #parPF.N
  ## nM is the no MCMC steps synthesized to make up one mutation step
  ## nE is the number of edges randomly selected and flipped to make up
  ## the proposal for each MCMC step in the mutation

  Gc = copy(G); logl_GGc = copy(logl_GG); logl_GYc = copy(logl_GY)

  Gpr = falses(N, p, p); logl_GYpr = zeros(N); logl_GGpr = zeros(N)

  log_acc = zeros(N); decide = falses(N)

  for m in 1:nM

    Gpr = mutate_sym_prop(Gpr, Gc, hyper, parPF, pair)

    logl_GGpr = logp_GG(Gpr, Gp, hyper, parPF, is_Gp)

    logl_GYpr = logp_GY(Gpr, hyper, parPF, Yc)

    U = rand(N) #N independent samples i  (0,1)?

    ##

    log_acc = logl_GGpr .+ phi.*logl_GYpr

    log_acc = log_acc .- logl_GGc .- phi*logl_GYc

    decide = (log.(U) .<= log_acc)

    ##

    Gc = Gc + decide.*(Gpr .- Gc)

    logl_GGc = logl_GGc .+ decide.*(logl_GGpr .- logl_GGc)

    logl_GYc = logl_GYc .+ decide.*(logl_GYpr .- logl_GYc)

    ##

  end

  G = Gc
  #return(Gc)

end

#################################################
#################################################
# the proposal for the mutation of the Nxpxp graphs G
##G: Nxpxp, current graphs before mutation
##hyper: hyper-parameters
##parPF: particle filter parameters
##pair: is the object [[1,2],[1,3],...,[p-1,p]]

function mutate_sym_prop(Gpr, G, hyper, parPF, pair) # RUNS

  p = hyper.p; nE = parPF.nE;

  N = size(G)[1] #parPF.N

  Gpr = falses(N, p, p)

  for i in 1:N

    flip_edges = sample(pair, nE, replace=false)

    for j in 1:nE

      index1 = flip_edges[j][1]; index2 = flip_edges[j][2]

      Gpr[i, index1, index2] = true

    end

  end

  Gpr = Bool.(abs.(copy(G)-Gpr))

  return(Gpr)

end
