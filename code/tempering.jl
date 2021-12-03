using Statistics
using Distributions
using LinearAlgebra

include("./graph_decomposition.jl")
include("./normalising_constant_mc.jl")
include("./laplace.jl")
include("./graph_prior.jl")
include("./graph_mutate_mcmc.jl")
include("./likelihood.jl")



#############################################################
##S: change points on 1:Tau; either "nothing" or vector of integers
##hyper: model hyper-parameters
##Y: data in a Tau*p matrix
##parPF: parameters for PF
##pair: is the object [[1,2],[1,3],...,[p-1,p]]

function test_logl_est(S, hyper, Y, parPF, pair, G, Gp, logw, w_std, logl_GG, logl_GY, Gc, logl_GGc, logl_GYc, logl_GGpr, logl_GYpr, log_acc, decide)

  Tau = hyper.Tau; p = hyper.p; N = parPF.N; k = Int64 ## k could be 0

  if (isnothing(S)==1) k = 0 else k = length(S) end

  temp = Array{Any}(nothing, k+1)
  ## temp: the temperatures on the fly

  G = falses(N, p, p)
  ## adjacency matrix: Nxpxp

  Gp = falses(N, p, p)
  ## parents of particles G through tempering steps: Nxpxp
  Gc = falses(N, p, p) #for mutation step

  nc = 1; nn = Tau ## current & next data position in 1:Tau

  logw = zeros(N); w_std = zeros(N)

  logl_GG = zeros(N); logl_GY = zeros(N)

  logl_GGc = zeros(N); logl_GYc = zeros(N); logl_GGpr = zeros(N); logl_GYpr = zeros(N)

  log_acc = zeros(N); decide = falses(N)

  ########################################
  ########################################
  for l in 1:(k+1) ## Main Loop Begins: k>=0
  ########################################
  ########################################
    if (l==1) G = sample_G(G, hyper, parPF)

    else  G = sample_GG(G, Gp, hyper, parPF)

    end

    if (l==(k+1)) nn = Tau else nn = S[l]-1 end

    Yc = Y[nc:nn,:]; is_Gp = (l>1)

    phi_c = 0.0; phi_n = 0.0

    ############################
    ############################
    while (phi_n < 0.999999) ## Tempering Loop begins
    ############################
    ############################

      phi_c = phi_n

      logl_GY = logp_GY(G, hyper, parPF, Yc)

      phi_n = tempering(logw, logl_GY, parPF, phi_c)

      logw = logw .+ (phi_n-phi_c).*logl_GY

      ###################
      ###################
      if (phi_n < 0.999999) ## necessarily resample + mutate
      ###################
      ###################

        if (isnothing(temp[l])) temp[l] = phi_n

        else temp[l] = [temp[l]; phi_n]

        end

        ##########
        ##resample

        w_std = exp.(logw.-maximum(logw))

        index = rand(Categorical(w_std./sum(w_std)), N)

        G = G[index,:,:]; logl_GY = logl_GY[index]

        if (l>1) Gp = Gp[index,:,:] end

        ##########
        ##mutate

        logl_GG = logp_GG(G, Gp, hyper, parPF, is_Gp)

        G = mutate(Gc, G, Gp, phi_n, hyper, parPF, logl_GY, logl_GG, logl_GGc, logl_GYc, logl_GGpr, logl_GYpr, Yc, is_Gp, pair, log_acc, decide)

        logw = zeros(N)

        ##########

      end
      # end of if (phi_n < 0.999999):
      # necessarily have resampled + mutated (equalised weights)
      ###################
      ###################

    end
    ## end of while (phi_n < 0.999999)
    ## at this position we have weighted particles
    ## updates of critical parameters
    ################
    ################

    nc = nn + 1; Gp = G

  end
  ## end of loop over change points
  ############################
  ############################

  return(temp)

end

#################################################
#################################################
# applies bisection method to obtain the temperature needed
# for a given target ESS
##logw: current log-weights before addition of extra log-like
##logl_G: (phi-phi_c)*logl_G is extra log-likelihood weighting
##parPF: PF algorithmic parameters
##phi_c: current temperature

function tempering(logw, logl_G, parPF, phi_c) # RUNS

  N = parPF.N; ESS0 = parPF.ESS0; TOL = parPF.TOL

  phiL = phi_c; phiR = 1.0; temp = 1.0

  ESS_R = compute_ESS(logw + ((phiR-phi_c).*logl_G))

  if (ESS_R < ESS0)

    while (phiR-phiL>TOL)

      temp = (phiL+phiR)/2.0

      ESS_C = compute_ESS(logw + ((temp-phi_c).*logl_G))

      if (ESS_C<ESS0) phiR = temp else phiL = temp end

    end

  end

  return(temp)

end

#################################################
#################################################
# calculation of ESS in [0,1] given log-weights
##logW: the vector of log-weights

function compute_ESS(logW) # RUNS

  N = length(logW)

  logWC = logW .- maximum(logW) # centralise weights

  WC = exp.(logWC)

  ESS = ((sum(WC))^2)/(sum(WC.^2))

  return(ESS/N)

end
