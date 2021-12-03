## Last Update: 16 Sep 2020

using Distributions
using Plots
using DelimitedFiles
using LinearAlgebra
using LightGraphs
using GraphPlot
using SpecialFunctions
using DataFrames
using CSV
include("./particle_MCMC.jl")

########################################################
struct hyper_pars
    p::Int ## no of nodes
  Tau::Int ## no of time instances
    w::Float64 ## w in (0,1], gives E[no of edges]=p*w for G0
    z::Float64 ## z in (0,1], gives E[no of edges flips]=p*z for G|Gp
delta::Float64 ## delta>2 is parameter of G-Wishart(delta, D)
  tau::Float64 ## tau>0 is s.t. D=tau*I in G-Wishart(delta,D)
  theta::Float64 # in [0,1], parameter of Geometric prior
end
#############################
struct MCMC_pars
      M::Int ## no of MCMC steps
     pB::Float64 ## min p(Birth)
end
#############################
struct PF_pars
    N::Int ## no of particles
 ESS0::Float64 ## ESS0 in (0,1) is threshold for ESS triggering resampling
  TOL::Float64 ## tolerance level for bisection method convergence
   nE::Int ## no of edges flipped at mutation step
   nM::Int ## no of single MCMC steps synthesized to make up a mutation step
   num_it::Int ## no of MC interations for normalising constant
   Laplace::Bool ## true = use the Laplace method for computing norm const
end
#############################
struct Tree
    G_yes::Array ## Sub graphs for decomposable components
    D_yes::Array ## Sub matrices of D for decomposable components
    G_no::Array ## Sub graphs for non-decomposable components
    D_no::Array ## Sub matrices of D for decomposable components
    G_sep::Array ## Sub graphs for decomposable separators
    D_sep::Array ## Sub matrices of D for decomposable separators
end
#############################
#############################

#### testing the code

hyper = hyper_pars(10, 100, 400.0, 0.8, 1.0, 1.0, 3.0, 1.0/3.0, 0.5)

parPF = PF_pars(10, 0.7, 0.001, 3, 1, 10, true)

parMCMC = MCMC_pars(10, 0.5)

### Data
#Y = rand(Normal(), hyper.Tau, hyper.p)

#Y = CSV.read("datamonthly.csv")
#Y = Array(Y[!,2:end])

#Y = CSV.read("generated_data_9_24_36_54_68_78_88.csv")
#Y = Array(Y[!,2:end])

#Y = CSV.read("generated_data_no_cp.csv")
#Y = Array(Y[!,2:end])

Y = CSV.read("generated_data_random_normal.csv")
Y = Array(Y[!,2:end])
###

#G = sample_G(hyper, parPF)

#T = decompose_G(G[1, :, :], Diagonal(ones(10)))

#logp_GY(G, hyper, parPF, Y)

########
#using Profile
#@profile pseudo_MCMC([50], hyper, Y, parMCMC, parPF)
out = pseudo_MCMC([50], hyper, Y, parMCMC, parPF)
#Juno.profiler()
#Profile.print(mincount=1000)
########

########
#@time out = pseudo_MCMC([50], hyper, Y, parMCMC, parPF)
########

########################################
########################################
########################################
########################################

#x = rand(1000)
#y = rand(1000)

#function sum_diff_mine(x,y)
#     sumx = sum_mine(x)
#     sumy = sum_mine(y)
#     return(sumx-sumy)
#end

#function sum_mine(x)
#    return(sum(x))
#end

#res = @time sum_diff_mine(x,y)
#@time sum_diff_mine(x,y)

#Juno.@enter sum_diff_mine(x,y)

########################################
########################################

#g = DiGraph(4)
#gplot(g, nodelabel=1:4)

#gplot(Graph(Symmetric(G[1,:,:])), nodelabel=1:5)

#Int.(G)

########################################
########################################
########################################
########################################
