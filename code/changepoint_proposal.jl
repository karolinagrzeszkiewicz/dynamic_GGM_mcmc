using Statistics
using Distributions
using LinearAlgebra

include("./helperfunctions.jl")

#################################################
#################################################
# samples the proposal for change point given current S;
# returns proposal and type, (B, D, M)
##S: current position of change points
##m: mth MCMC step
##hyper: model hyper-parameters
##parMCMC: MCMC parameters


function proposal(S, m, hyper, parMCMC)

  Tau = hyper.Tau; p = hyper.p; pB = parMCMC.pB; k = Int64; Spr = nothing

  if (isnothing(S)) k = 0 else k = length(S) end

  pr_type = String

  if isodd(m) || k == 0

        empty = get_empty_list(S, Tau, p)
        n_empty = length(empty)

        if k == 0 && n_empty > 0
            pr_type = "Birth"
        elseif n_empty == 0
            pr_type = "Death"
        else
            U = rand()
            if U < pB
                pr_type = "Birth"
            else
                pr_type = "Death"
            end
        end
   else
        pr_type = "Move"

   end

  Spr = sample_MCMC_SS(S, pr_type, hyper)

  return(Spr, pr_type)

end




#################################################
#################################################
# samples the proposal for change point given current S and
# given the type of proposal in pr_type;
# returns proposal
##S: current position of change points
##pr_type: the type of proposal: "Birth", "Death" or "Move"
##hyper: model hyper-parameters

function sample_MCMC_SS(S, pr_type, hyper)

  Tau = hyper.Tau; p = hyper.p; k = Int64; Spr = nothing

  if (isnothing(S)) k = 0 else k = length(S) end

  ###########################
  ###########################
  if (pr_type == "Birth")
  ###########################

        empty = get_empty_list(S, Tau, p)

        l = length(empty)

        if l == 0

            error("Birth not possible: there is no space for a new changepoint")

        else

            Unif = DiscreteUniform(1, l)
            sBidx = rand(Unif)
            sB = empty[sBidx]
            Spr = append_changepoint(S, sB)
        end


  ###########################
  elseif (pr_type == "Death")  ## certain that k>=1

    if k == 0
        error("Death not possible: there are no changepoints to kill")
    else
        Unif = DiscreteUniform(1, k)
        sDidx = rand(Unif)
        sD = S[sDidx]
        Spr = delete_changepoint(S, sD)
    end


   ###########################
  else ## certain that k>=1, Move

      # delete changepoint
      Unif = DiscreteUniform(1, k)

      sMidx = rand(Unif)
      sM = S[sMidx]
      SsM = delete_changepoint(S, sM)

      # add changepoint - random walk
      direction_right = (rand(Uniform()) > 0.5)

      Geom = Geometric(hyper.g)
      numSteps = rand(Geom) + 1
      numSteps = if direction_right numSteps else -numSteps end

      sB = sM + numSteps

      Spr = append_changepoint(SsM, sB)

  end
  ###########################
  ###########################

  return(Spr)

end



#TO DO

#################################################
#################################################
# calculates log of acceptance probability for proposal
##S: current position of change points
##Spr: proposed position of change points
##pr_type: type of proposal, B, D or M
##hyper: model hyper-parameters
##parMCMC: MCMC parameters
##logl_pr: estimate of log of p(Y|Sp)
##logl_c: estimate of log of p(Y|S)

function log_acc_prob(S, Spr, pr_type, hyper, logl_pr, logl_c)

  target_ratio = logl_pr - logl_c + logp_S(Spr, hyper) - logp_S(S, hyper) #RATIO OF TARGETS

  transition_kernel = 0.0
  reversed_transition_kernel = 0.0
  ratio = 0.0

  if (pr_type == "Birth")

    reversed_transition_kernel = logq_MCMC_SS(Spr, S, "Death", hyper)

    transition_kernel = logq_MCMC_SS(S, Spr, "Birth", hyper)


  elseif (pr_type == "Death")

      reversed_transition_kernel = logq_MCMC_SS(Spr, S, "Birth", hyper)

      transition_kernel = logq_MCMC_SS(S, Spr, "Death", hyper)

  else

    reversed_transition_kernel = logq_MCMC_SS(Spr, S, "Move", hyper)

    transition_kernel  = logq_MCMC_SS(S, Spr, "Move", hyper)

  end

  transition_ratio = reversed_transition_kernel - transition_kernel

  return(target_ratio + transition_ratio)

end


#################################################
#################################################
# calculates the log-transition kernel for the proposal given current S and
# given the type of proposal in pr_type;
##S: current position of change points
##Sn: next position of change points
##pr_type: the type of proposal: "Birth", "Death" or "Move"
##hyper: model hyper-parameters
##parMCMC: MCMC parameters
##lNBs: the log-pdf of Neg.Bin on 0:Tau-1

function logq_MCMC_SS(S, Sn, pr_type, hyper)

  Tau = hyper.Tau; k = Int64; kn = Int64; result = 0.0

  if (!isnothing(S)) k = length(S) else k = 0 end

  if (!isnothing(Sn)) kn = length(Sn) else kn = 0 end

  ##########################
  ##########################
  if (pr_type == "Birth") ## kn>=1, k>=0, kn=k+1

    all_same = true; i = 1; sample = Int64;

    while (all_same && i <= k)

      all_same = (S[i]==Sn[i]); i = i+1

    end

    if (!all_same)

      sample = Sn[i-1];


    else

      sample = Sn[i];

    end

    result = logqB(sample, S, hyper)

  ##########################
  ##########################
  elseif (pr_type == "Death") ## kn>=0, k>=1, kn=k-1

    all_same = true; i = 1; sample = Int64

    while (all_same && i <= kn)

      all_same = (S[i]==Sn[i]); i = i+1

    end

    if (!all_same) sample = S[i-1] else sample = S[i] end

    result = logqD(sample, S, hyper)

  ##########################
  ##########################
  else # norm. constant not needed

      function not_in_Sn(x)
          return(!(issubset(x, Sn)))
      end

      function not_in_S(x)
          return(!(issubset(x, S)))
      end

      s_pr_idx = findall(not_in_S, Sn)[1]
      s_idx = findall(not_in_Sn, S)[1]

    #if !(length(s_pr_idx) > 0 && length(s_idx) > 0)
    #  s_pr_idx = 1
    #  s_idx = 1
    #end

    s_pr = Sn[s_pr_idx]
    s = S[s_idx]

    result = logqM(s_pr, s, S, hyper)

  end
  ##########################
  ##########################

  return(result)

end



#################################################
#################################################
# calculates the log-transition kernel for the proposal given current S,
# and given that the proposal corresponds to "Birth"
##sample: the relevant random sample from Neg.Bim.
##sL: the left point for the support of sample
##sR: the right point for the support of sample
##lNBs: the log-pdf of Neg.Bin on 0:Tau-1

#TO DO: ADD logpB
# q(S_{1:k+1}', k+1 | S_{1:k}, k) = p(B|k)*q(S*_{k+1}| S_{1:k}, B)
#sample is S*_{k+1}

function logqB(sample, S, hyper)

    Tau = hyper.Tau; k = Int64; p = hyper.p

    if (!isnothing(S)) k = length(S) else k = 0 end

    result = 0.0

    if is_valid(sample, S, Tau, p)
        choices_for_sample = find_empty(S, Tau, p)
        log_pB = logpB(S, Tau, p)
        if choices_for_sample > 0
            result = log(1/choices_for_sample) + log_pB
        else
            result = log(0)
        end
    else
        result = log(0)
    end

  return(result)

end

function logpB(S, T, p)

    k = Int64

    if (!isnothing(S)) k = length(S) else k = 0 end

    res = 0.0

    if k == 0
        res = log(1)
    else
        empty_spots = find_empty(S, T, p)
        if empty_spots > 0
            res = log(0.5)
        else
            res = log(0)
        end
    end

    return(res)

end


#################################################
#################################################
# calculates the log-transition kernel for the proposal given current S,
# and given that the proposal corresponds to "Death"
##sample: the relevant random sample from Neg.Bim.
##S: the current change points
##lNBs: the log-pdf of Neg.Bin on 0:Tau-1

function logqD(sample, S, hyper)

    Tau = hyper.Tau; k = Int64; p = hyper.p

    k = Int64

    if (!isnothing(S)) k = length(S) else k = 0 end

    result = 0.0

    if k == 0
        result = log(0)
    else
        log_pB = logpD(S, Tau, p)
        result = log(1.0/k) + log_pB
    end

  return(result)

end

#proposal probability of death

function logpD(S, T, p)

    k = Int64

    if (!isnothing(S)) k = length(S) else k = 0 end

    res = 0.0

    if k == 0
        res = log(0)
    else
        empty_spots = find_empty(S, T, p)
        if empty_spots > 0
            res = log(0.5)
        else
            res = log(1)
        end
    end

    return(res)
end

#################################################
#################################################
# calculates the log-transition kernel for the proposal given current S,
# and given that the proposal corresponds to "Move"
##sample: the relevant random sample from Neg.Bim.
##sL: the left point for the support of sample
##sR: the right point for the support of sample
##lNBs: the log-pdf of Neg.Bin on 0:Tau-1

function logqM(s_pr, s, S, hyper)

    Tau = hyper.Tau; k = Int64; p = hyper.p

    if (!isnothing(S)) k = length(S) else k = 0 end

    result = 0.0

    if k == 0
        result = log(0)
    else

        dist = s_pr - s - 1
        Geom = Geometric(hyper.g)
        prob_dist = logpdf(Geom, dist)

        result = log(0.5) + prob_dist

    end

    return(result)

end
