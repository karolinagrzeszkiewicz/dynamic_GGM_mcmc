using Statistics
using LinearAlgebra
using Distributions

#################################################
#################################################
# calculates the log of the transition probability
# of p(G|Gp) if is_GP = 1, otherwise it calculates the log of p(G),
# - in the latter case Gp = zeros(N,p,p).
##G: the phaphs, Nxpxp
##Gp: the parents of the graphs, Nxpxp
##hyper: hyper-parameters
##parPF: Particle filter parameters
##isGp: 1 if there are parents; 0 otherwise

function logp_GG(G, Gp, hyper, parPF, is_Gp) # OK 17/07

  p = hyper.p; w = hyper.w; z = hyper.z; N = parPF.N

  N_edges = Int64(p*(p-1)/2)

  prob = 0.0; sumG = zeros(N)

  if (is_Gp==0)

    prob = 2.0*w/(p-1.0)

    sumG = (sum(G, dims=[2,3]))[1:N,1,1]

  elseif (is_Gp==1)

    prob = 2.0*z/(p-1.0)

    sumG = (sum(abs.(G-Gp), dims=[2,3]))[1:N,1,1]

  end

  logb = [ log(binomial(N_edges,i)) for i in sumG ]

  return(logb + sumG.*log(prob) + (N_edges.-sumG).*log(1.0-prob))

end

#################################################
#################################################
# samples from transition of signal model
# stored in upper triangular matrix (i.e. adjacency matrix)
##Gp: parent particles
##hyper: hyper-parameters
##parPF: Particle filter parameters

function sample_GG(G, Gp, hyper, parPF, N_per_rank = 1, is_master = false)

  N = if parPF.is_parallel N_per_rank else parPF.N end
  N_graphs = if (parPF.is_parallel && !is_master) N_per_rank else parPF.N end

  p = hyper.p; z = hyper.z

  G = falses(N_graphs, p, p)

  dist = Bernoulli(2.0*z/(p-1)) # average no of nodes selected z*p

  lv = Int64(p*(p-1)/2); vector_G = rand(dist, N, lv)
  # N x lv array

  l = 0

  for i in 1:(p-1)

    l = Int64((i-1.0)*(p-i/2.0)) # CHECK THIS!!!!

    for n in 1:N

      G[n,i,(i+1):p] = abs.(Gp[n,i,(i+1):p] - vector_G[n,(l+1):(l+p-i)] )

    end

  end

  return(G)

end

#################################################
#################################################
# samples from prior for G at time 1,
# stored in upper triangular matrix (i.e. adjacency matrix)
##hyper: hyper-parameters
##parPF: Particle filter parameters

function sample_G(G, hyper, parPF, is_parallel = false, N_per_rank = 1, is_master = false)

  N = if parPF.is_parallel N_per_rank else parPF.N end
  N_graphs = if (parPF.is_parallel && !is_master) N_per_rank else parPF.N end

  p = hyper.p; w = hyper.w; G = falses(N_graphs, p, p)

  dist = Bernoulli(2.0*w/(p-1)) # mean no of edges is w*p

  lv = Int64(p*(p-1)/2); vector_G = rand(dist, N, lv)

  l = 0

  for i in 1:(p-1)

    l = Int64((i-1.0)*(p-i/2.0))

    for n in 1:N

      G[n,i,(i+1):p] = copy(vector_G[n,(l+1):(l+p-i)])

    end

  end

  return(G)

end
