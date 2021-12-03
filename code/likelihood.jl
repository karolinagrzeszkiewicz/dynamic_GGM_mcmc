using Statistics
using LinearAlgebra
using Distributions


# calculates the log of the transition probability
# of p(Yc|G) for the given set of data points Yc
##G: Nxpxp upper triangular adgacency matrix representing graphs
##hyper: hyper-parameters
##parPF: Particle filter parameters
##Yc: currect set of data

function logp_GY(G, hyper, parPF, Yc)

  n = size(Yc)[1]; p = hyper.p; delta = hyper.delta

  tau = hyper.tau; num_it = parPF.num_it

  use_laplace = parPF.Laplace

  Sc = transpose(Yc)*Yc

  log_const = (-n*p/2.0)*log(2.0*pi)

  log_posterior = log_Iota(G, delta+n,  tau*Matrix(1.0I, p, p)+Sc, num_it, use_laplace)

  #Diagonal(tau*ones(p))
  log_prior = log_Iota(G, delta, tau*Matrix(1.0I, p, p)+zeros(p,p), num_it, use_laplace)

  log_G = log_const .+ log_posterior .- log_prior

  #return(log_const .+ log_G)
  return(log_G)

end

# returns the log of the normalising constant
# for graphs G
##G: Nxpxp; all N current graphs represented by pxp adjacency matrices
##D: positive-definite, symmetric matrix
##delta: delta>2

function log_Iota(G, delta, D, num_it, use_laplace)

  N = size(G)[1]; p = size(G)[2]

  logI = zeros(N)

  for n in 1:N

      tree = decompose_G(G[n,:,:], D)


      logI1 = logIyes(tree.G_yes, delta, tree.D_yes)

      logI2 = logIno(tree.G_no, delta, tree.D_no, num_it, use_laplace)

      logI3 = logIyes(tree.G_sep, delta, tree.D_sep)

      logI[n] = logI1 + logI2 - logI3

   end

  return(logI)

end


################################################
################################################
# decomposes single graph G into primary components,and separators;
# it returns separately the cliques, the non-decomposable primary
# components and the separators; it also returns separately the
# corresponding sub-matrices of D that are relevant for each
# of the identified sub-graphs
##G: single graph G
##D: positive-definite, symmetric

function decompose_G(G, D)

  Gadj = Int.(Symmetric(G))
  # this is added to ensure combatability with the functions below

  decomp = graph_decomposition.decompose_graph(Gadj)

  is_chordal = decomp[1]; complete = decomp[2]

  incomplete = decomp[3]; separators = decomp[4]

  if is_chordal

    D_yes = [D[c,c] for c in complete]

    G_yes = [G[c,c] for c in complete]

    D_sep = [D[s,s] for s in separators]

    G_sep = [G[s,s] for s in separators]

    D_no = []; G_no = []

  else

    D_yes = [D[c,c] for c in complete]

    G_yes = [G[c,c] for c in complete]

    D_sep = [D[s,s] for s in separators]

    G_sep = [G[s,s] for s in separators]

    D_no = [D[c,c] for c in incomplete]

    G_no = [G[c,c] for c in incomplete]

  end

  T = Tree(G_yes, D_yes, G_no, D_no, G_sep, D_sep)

  return(T)

end

# returns the exact log of the normalising constant for complete sub-graphs,
##G: collection of complete sub-graphs
##D: corresponding collection of sub-matrices out of main matrix parameter
##delta: parameter, >2

function logIyes(G, delta, D)

  nT = length(G)

  result = 0.0

  for i in 1:nT  ## nT could be 0

    p = size(G[i])[1]

    term1 = (p*(delta+p-1.0)/2.0)*log(2)

    term2 = loggengamma((delta+p-1.0)/2.0, p)

    term3 = ((delta+p-1.0)/2.0)*logdet(D[i])

    result = result + (term1+term2-term3)

  end

  return(result)

end
################################################
################################################
# returns the log of the generalised gamma distribution
# formula from Lenkoski and Dobra
function loggengamma(a, d)

  term1 = (d*(d-1.0)/4.0)*log(pi)

  term2 = sum([loggamma(a-((i-1.0)/2.0)) for i in 1:d])

  if a > (d-1.0)/2.0
      return(term1+term2)
  else
      return(log(0))
  end

end

function logIno(G, delta, D, num_it, use_laplace)

  nT = length(G) #works

  result = 0.0

  #use_laplace = (delta > 50)

  #use_laplace = true

  for i in 1:nT  ## nT could be 0

    if use_laplace
        result = result + log_IG_laplace(G[i], delta, Symmetric(D[i]))
    else
        result = result + normalising_constant_mc(G[i], delta, Symmetric(D[i]), num_it)
    end

  end

  return(result)

end

###############################################
###############################################
