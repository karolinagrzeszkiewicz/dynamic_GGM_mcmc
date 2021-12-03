using Statistics
using Distributions
using LinearAlgebra

include("./graph_decomposition.jl")
include("./normalising_constant_mc.jl")
include("./laplace.jl")
include("./graph_prior.jl")
include("./graph_mutate_mcmc.jl")
include("./changepoint_prior.jl")
include("./changepoint_proposal.jl")
include("./likelihood.jl")
include("./tempering.jl")
include("./particle_filter.jl")


#############################
struct hyper_pars
    p::Int ## no of nodes
  Tau::Int ## no of time instances
    w::Float64 ## w in (0,1], gives E[no of edges]=p*w for G0
    z::Float64 ## z in (0,1], gives E[no of edges flips]=p*z for G|Gp
delta::Float64 ## delta>2 is parameter of G-Wishart(delta, D)
  tau::Float64 ## tau>0 is s.t. D=tau*I in G-Wishart(delta,D)
theta::Float64 # in [0,1], parameter of Geometric prior
    g::Float64 #hyperparameter for geometric proposal
end
#############################
struct MCMC_pars
      M::Int ## no of MCMC steps
     pB::Float64 ## p(Birth)
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
   is_parallel::Bool
   master :: Int
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

#################################################
#################################################
##S0: initial position of change points on 1:Tau
##hyper: hyper-parameters
##Y: data in a pxTau matrix
##parMCMC: MCMC parameters
##parPF: PF parameters

function pseudo_MCMC(S0, hyper, Y, parMCMC, parPF)

  p = hyper.p; Tau = hyper.Tau; M = parMCMC.M; N = parPF.N

  S_all = Array{Any}(nothing, M+1)
  G_all = Array{Any}(nothing, M+1)

  final_graphs = zeros(Tau,p,p)

  if (isnothing(S0) == 1) k0 = 0 else k0 = length(S0) end

  acc = falses(M)

  pair = make_pairs(hyper)

  #####
  #####

  Sc = S0
  Spr = S0

  #println("Sc = ", Sc)

  #create memory to be passed to the particle filter:

  G = falses(N, p, p)

  Gp = falses(N, p, p)

  Gc = falses(N, p, p)

  logw = zeros(N); w_std = zeros(N)

  max_logw = 0.0; shifted_w = exp.(logw)

  logl_GG = zeros(N); logl_GY = zeros(N)

  logl_GGc = zeros(N); logl_GYc = zeros(N); logl_GGpr = zeros(N); logl_GYpr = zeros(N)

  log_acc = zeros(N); decide = falses(N) #for mutation

  particle_history = zeros(k0+1,N,p,p) #problem: k changes from iteration to iteration

  logl_YG_history = zeros(k0+1, N)

  particle_weights_final = zeros(N)

  G_mean = zeros(k0+1,p,p)

  cur_mean = zeros(p,p)


  temps = test_logl_est(Sc, hyper, Y, parPF, pair, G, Gp, logw, w_std, logl_GG, logl_GY, Gc, logl_GGc, logl_GYc, logl_GGpr, logl_GYpr, log_acc, decide)
  ## pseudo-run, temps: temperatures

  (logl_c, Gc) = logl_est(Sc, hyper, Y, parPF, temps, pair, G, Gp, logw, w_std, max_logw, shifted_w, logl_GG, logl_GY, particle_history, logl_YG_history, particle_weights_final, G_mean, cur_mean, Gc, logl_GGc, logl_GYc, logl_GGpr, logl_GYpr, log_acc, decide)
  (logl_pr, G_pr) = logl_est(Sc, hyper, Y, parPF, temps, pair, G, Gp, logw, w_std, max_logw, shifted_w, logl_GG, logl_GY, particle_history, logl_YG_history, particle_weights_final, G_mean, cur_mean, Gc, logl_GGc, logl_GYc, logl_GGpr, logl_GYpr, log_acc, decide)
  ## initialise log-likelihood estimate
  #println("logP(Y|Sc) = ", logl_c)


  S_all[1] = Sc # invoke using S_all[i][j]

  G_all[1] = Gc


  ##############################
  ##############################
  for m in 1:M

    println("PMCMC iteration m = ", m)

  ##############################
  ##############################
    (Spr, pr_type) = proposal(Sc, m, hyper, parMCMC)
    println("proposal Spr = ", Spr)
    println("proposal type = ", pr_type)

    if (exp(logp_S(Spr, hyper)) > 0.0) #otherwise if the prior is 0 for Spr then we reject it immediatley

    #temps = test_logl_est(Spr, hyper, Y, parPF)
        temps = test_logl_est(Spr, hyper, Y, parPF, pair, G, Gp, logw, w_std, logl_GG, logl_GY, Gc, logl_GGc, logl_GYc, logl_GGpr, logl_GYpr, log_acc, decide)

        #logl_pr = logl_est(Spr, hyper, Y, parPF, temps)
        (logl_pr, G_pr) = logl_est(Spr, hyper, Y, parPF, temps, pair, G, Gp, logw, w_std, max_logw, shifted_w, logl_GG, logl_GY, particle_history, logl_YG_history, particle_weights_final, G_mean, cur_mean, Gc, logl_GGc, logl_GYc, logl_GGpr, logl_GYpr, log_acc, decide)

        println("logP(Y|Spr) =", logl_pr)

        log_acc_p = log_acc_prob(Sc, Spr, pr_type, hyper, logl_pr, logl_c)
        println("log acceptance probability = ", log_acc_p)

        U = rand()

        if (log(U) < log_acc_p)

          #Sc = Sp; logl_c = logl_pr
          Sc = Spr; logl_c = logl_pr; Gc = G_pr

          acc[m] = 1

          println("accepted!")
          #println("rejected")

        end

    end

    S_all[m+1] = Sc
    G_all[m+1] = Gc

    #add to final graphs
    Sk_prev = 1
    K = length(Sc)
    for k in 1:K
        Sk = Sc[k]
        final_graphs[Sk_prev:(Sk-1),:,:] += permutedims((repeat(Gc[k,:,:], outer = [1,1,Sk-Sk_prev])), [3,2,1])
        Sk_prev = Sk
    end
    final_graphs[Sk_prev:Tau,:,:] += permutedims((repeat(Gc[K+1,:,:], outer = [1,1,Tau-Sk_prev+1])), [3,2,1])


  end

  final_graphs = final_graphs/M
  ##############################
  ##############################
  return(S_all, final_graphs, G_all, acc)


end

function make_pairs(hyper)  # RUNS

  p = hyper.p; N_edges = Int64(p*(p-1)/2)

  pair = [fill(1, 1) for i in 1:N_edges]

  sum = 0

  for i in 1:(p-1)

    for j in (i+1):p

      pair[sum+j-i] = [i,j]

    end

    sum = sum + p-i

  end

  return(pair)

end
