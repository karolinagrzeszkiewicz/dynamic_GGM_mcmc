using Statistics
using Distributions
using LinearAlgebra

include("./graph_decomposition.jl")
include("./normalising_constant_mc.jl")
include("./laplace.jl")
include("./graph_prior.jl")
include("./graph_mutate_mcmc.jl")
include("./likelihood.jl")


#################################################
#################################################
##S: change points on 1:Tau
##hyper: model hyper-parameters
##Y: data in a Tau*p matrix
##parPF: parameters for PF
##temp: tempetatured as obtained via test_logl_est function
##pair: is the object [[1,2],[1,3],...,[p-1,p]]

function logl_est(S, hyper, Y, parPF, temp, pair, G, Gp, logw, w_std, max_logw, shifted_w, logl_GG, logl_GY, particle_history, logl_YG_history, particle_weights_final, G_mean, cur_mean, Gc, logl_GGc, logl_GYc, logl_GGpr, logl_GYpr, log_acc, decide)

  Tau = hyper.Tau; p = hyper.p; N = parPF.N; k = Int64 ## k could be 0

  if (isnothing(S)==1) k = 0 else k = length(S) end

  #reset relevant matrices

  G = falses(N, p, p)
  ## adjacency matrix: Nxpxp

  Gp = falses(N, p, p)
  ## parents of particles G through tempering steps: Nxpxp
  Gc = falses(N, p, p) #for mutation step

  logw = zeros(N); w_std = zeros(N)

  max_logw = 0.0; shifted_w = exp.(logw)

  logl_GG = zeros(N); logl_GY = zeros(N)

  logl_GGc = zeros(N); logl_GYc = zeros(N); logl_GGpr = zeros(N); logl_GYpr = zeros(N)

  log_acc = zeros(N); decide = falses(N)

  particle_history = zeros(k+1,N,p,p)

  logl_YG_history = zeros(k+1, N)

  particle_weights_final = zeros(N)

  G_mean = zeros(k+1,p,p)

  cur_mean = zeros(p,p)

  nc = 1; nn = Tau ## current & next data position in 1:Tau

  result = 0.0

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

    ####################

    tempc = temp[l]

    if (isnothing(tempc)) tempc = 1.0

    else tempc = [tempc; 1.0]

    end

    t = length(tempc); phi_c = 0.0; phi_n = 0.0

    ############################
    ############################
    for j in 1:t ## loop over temperatures; t>=1 as it includes 1.0
    ############################
    ############################

      phi_c = phi_n

      logl_GY = logp_GY(G, hyper, parPF, Yc)

      phi_n = tempc[j]

      logw = logw .+ (phi_n-phi_c).*logl_GY

      ###################
      ###################
      if (phi_n < 0.999999) ## necessarily resample + mutate + estimate
      ###################
      ###################

        ##########
        ##estimate

        max_logw = maximum(logw)

        shifted_w = exp.(logw .- max_logw)

        result = result + max_logw + log(sum(shifted_w)) - log(N)

        ##########
        ##resample

        w_std = shifted_w./sum(shifted_w)

        index = rand(Categorical(w_std), N)

        G = G[index,:,:]; logl_GY = logl_GY[index]

        if (l>1) Gp = Gp[index,:,:] end

        if l > 1
            particle_history[1:(l-1), :,:,:] = particle_history[1:(l-1), index,:,:]
            logl_YG_history[1:(l-1),:] = logl_YG_history[1:(l-1),index]
        end

        ##########
        ##mutate

        logl_GG = logp_GG(G, Gp, hyper, parPF, is_Gp)

        #in-place modification of G
        mutate(Gc, G, Gp, phi_n, hyper, parPF, logl_GY, logl_GG, logl_GGc, logl_GYc, logl_GGpr, logl_GYpr, Yc, is_Gp, pair, log_acc, decide)

        logw = zeros(N)


        ##########

      end
      # end of if (phi_n < 0.999999):
      # necessarily have resampled + mutated (equalised weights)
      ###################
      ###################

    end
    #end of loop over temperatures
    #we need an estimate for the last temperature phi_n = 1


    #estimate

    max_logw = maximum(logw)

    shifted_w = exp.(logw .- max_logw)

    result = result + max_logw + log(sum(shifted_w)) - log(N)

    #weights go to zero after estimating (so that the stimation step is not duplicated)
    logw = zeros(N)

    ################
    ################
    ## end of for j in 1:t
    ## at this position we have weighted particles
    ## updates of critical parameters

    nc = nn + 1; Gp = G

    #final particles for l
    particle_history[l,:,:,:] = Gp

    #likelihoods
    logl_YG_history[l,:] = logl_GY

  end
  ## end of loop over change points
  ############################
  ############################

  #mean graphs

  for i in 1:N
        particle_weights_final[i] = sum(logl_YG_history[:, i])
  end

  particle_weights_final = exp.(particle_weights_final .- maximum(particle_weights_final))
  particle_weights_final = particle_weights_final ./ sum(particle_weights_final)

  for i in 1:(k+1)
      weighted_G = particle_weights_final .* particle_history[i,:,:,:]
      cur_mean = sum(weighted_G[j,:,:] for j in 1:N)
      G_mean[i,:,:] = cur_mean
  end

  #mode (optional)
  #mode = argmax(particle_weights_std)
  #G_mode = particle_history[:,mode,:,:]

  return(result, G_mean)



end
