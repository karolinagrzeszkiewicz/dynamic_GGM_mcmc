using Statistics
using Distributions
using LinearAlgebra

include("./helperfunctions.jl")
#################################################
#################################################
# calculates log of prior prob for change points in S
##S: given position of change points
##hyper: model hyper-parameters

function logp_S(S, hyper) # RUNS

  result = logp_Sk(S, hyper) + logp_k(S, hyper)

  return(result)

end

# PRIOR ON NUMBER OF CHANGEPOINTS


function logp_k(S, hyper)

    theta = hyper.theta
    if (isnothing(S)==1) k = 0 else k = length(S) end

    Geom = Geometric(theta)
    result = logpdf(Geom, k)

    return(result)

end

#PRIOR ON CHANGEPOINTS GIVEN THEIR NUMBER


function logp_Sk(S, hyper)

    Tau = hyper.Tau; k = Int64; p= hyper.p

    if (isnothing(S)==1) k = 0 else k = length(S) end

    res = 0.0

    valid = are_valid(S, Tau, p)

    if valid
        n_permutations = get_permutations(Tau, p, k)
        res = log(1/n_permutations)
    else
        res = log(0)
    end

    return(res)

end
