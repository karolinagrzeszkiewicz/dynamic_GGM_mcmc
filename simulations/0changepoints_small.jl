#NULL SIMULATION – 0 changepoints, 50 data points generated from multivariate normal
#instructions: run julia simulations/0changepoints_small.jl from the dynamic_graphical_models_mcmc directory

using Statistics
using Distributions
using LinearAlgebra
using CSV
using DataFrames


curPath = split(pwd(), "/")
newPath = curPath[1:(length(curPath))]
setPath = join(newPath,"/")
cd(setPath)


include("../code/graph_decomposition.jl")
include("../code/normalising_constant_mc.jl")
include("../code/laplace.jl")
include("../code/graph_prior.jl")
include("../code/graph_mutate_mcmc.jl")
include("../code/likelihood.jl")
include("../code/helperfunctions.jl")
include("../code/tempering.jl")
include("../code/particle_filter.jl")
include("../code/particle_MCMC.jl")

p = 5 #number of nodes

hyper = hyper_pars(p, 100, 1.0, 1.0, 3.0, 1.0/3.0, 0.5, 0.25)
parPF = PF_pars(100, 0.7, 0.001, 3, 10, 10, true, false, 0)
parMCMC = MCMC_pars(100, 0.5)

curPath = split(pwd(), "/")
newPath = curPath[1:(length(curPath))]
setPath = join(newPath,"/")
cd(setPath)
path = pwd()


Id = Matrix(1.0I, p, p)
Y0 = zeros(50, p)
distr0 = MvNormalCanon(Id)

for i in 1:50
    Y0[i,1:p] = rand(distr0)
end


function run_PMCMC(Y0, hyper, parPF, parMCMC)

    (S, timeseries_graphs, G, acc) = pseudo_MCMC([], hyper, Y0, parMCMC, parPF)

    for i in 1:length(S)
        println("S for MCMC iteration = ",i," :", S[i])
    end

    for t in 1:hyper.Tau
        println("graph for t = ", t, " : ")
        println(timeseries_graphs[t,:,:])
    end

end

function get_time()
    @time run_PMCMC(Y0, hyper, parPF, parMCMC)
end

get_time()
