using Statistics
using Distributions
using LinearAlgebra
using MPI

include("./graph_decomposition.jl")
include("./normalising_constant_mc.jl")
include("./laplace.jl")
include("./graph_prior.jl")
include("./graph_mutate_mcmc.jl")
include("./likelihood.jl")

function parallel_PF(S, hyper, Y, parPF, temp, pair)

    MPI.Init()

    comm = MPI.COMM_WORLD

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    size = MPI.Comm_size(MPI.COMM_WORLD) #number of parallel proceses

    @assert mod(parPF.N, size) == 0

    N_per_rank = Int(parPF.N / size) #number of particles per rank, master rank has N though

    if (isnothing(S)==1) k = 0 else k = length(S) end

    model_data = init_model(hyper, parPF, N_per_rank, k, rank)

    filter_data = init_filter(hyper, parPF, N_per_rank, rank)

    nc = 1; nn = hyper.Tau

    for l in 1:(k+1)

        #sample the particles
        is_master = (parPF.master == MPI.Comm_rank(comm))

        if (l==1) model_data.G = sample_G(hyper, parPF, true, N_per_rank, is_master)

        else  model_data.G = sample_GG(model_data.Gp, hyper, parPF, true, N_per_rank, is_master)

        end

        if (l==(k+1)) nn = hyper.Tau else nn = S[l]-1 end

        Yc = Y[nc:nn,:]; is_Gp = (l>1)

        tempc = temp[l]

        if (isnothing(tempc)) tempc = 1.0

        else tempc = [tempc; 1.0]

        end

        t = length(tempc); phi_c = 0.0; phi_n = 0.0

        for j in 1:t ## loop over temperatures

            phi_c = phi_n

            model_data.logl_GY = logp_GY(model_data.G, hyper, parPF, Yc)

            phi_n = tempc[j]

            # log weights
            filter_data.weights = filter_data.weights .+ (phi_n-phi_c).* model_data.logl_GY

            if (phi_n < 0.999999) ## necessarily resample + mutate + estimate

                if parPF.master == MPI.Comm_rank(comm)

                    #MASTER RANK

                    # gather – master rank collects weights sent by other ranks
                    MPI.Gather!(MPI.IN_PLACE, UBuffer(filter_data.weights, N_per_rank), parPF.master, comm)

                    # normalise weights
                    # returns sum and max but normalises weights in-place
                    (sum_shifted_w, max_logw) = normalize_weights_and_get_sum_max(filter_data.weights)

                    # estimate likelihood
                    model_data.likelihood += max_logw + log(sum_shifted_w) - log(parPF.N)

                    # resample
                    resample(filter_data.resampling_indices, filter_data.weights)

                else

                    #OTHER RANKS

                    #gather but for other ranks i.e. send weights to be collected by master rank
                    MPI.Gather!(filter_data.weights, nothing, parPF.master, MPI.COMM_WORLD)

                end

                #ALL RANKS

                # broadcast resampled indices from master to other ranks
                MPI.Bcast!(filter_data.resampling_indices, parPF.master, MPI.COMM_WORLD)


                #copy states
                copy_states(model_data.G, filter_data.copy_buffer, filter_data.resampling_indices, rank, N_per_rank)

                # mutate
                mutate(model_data.G, model_data.Gp, phi_n, hyper, parPF, model_data.logl_GY, model_data.logl_GG, Yc, is_Gp, pair)

                #reset weights
                reset_weights(filter_data.weights)


            end #end of if (phi_n < 0.999999)



        end #end of loop over temperatures

        #we need an estimate for the last temperature phi_n = 1

        #estimate

        if parPF.master == MPI.Comm_rank(comm)

            #MASTER RANK

            # gather – master rank collects weights sent by other ranks
            MPI.Gather!(MPI.IN_PLACE, UBuffer(filter_data.weights, N_per_rank), parPF.master, comm)

            # normalise weights
            # returns sum and max but normalises weights in-place
            (sum_shifted_w, max_logw) = normalize_weights_and_get_sum_max(filter_data.weights)

            # estimate likelihood
            model_data.likelihood += max_logw + log(sum_shifted_w) - log(parPF.N)

        else

            #OTHER RANKS

            #gather but for other ranks i.e. send weights to be collected by master rank
            MPI.Gather!(filter_data.weights, nothing, parPF.master, MPI.COMM_WORLD)

        end

        #reset
        reset_weights(filter_data.weights)

        nc = nn + 1; model_data.Gp = copy(model_data.G)

        #gather particles in master rank to get particle_history[l,:,:,:]

        if parPF.master == MPI.Comm_rank(comm)
            MPI.Gather!(MPI.IN_PLACE, UBuffer(model_data.Gp, N_per_rank), parPF.master, comm)
            model_data.particle_history[l,:,:,:] = copy(model_data.Gp)
        else
            MPI.Gather!(model_data.Gp, nothing, parPF.master, comm)
        end

        #for gathered particles 1) compute logl_YG_history[l,:] oR 2) gather from model_data.logl_GY but 1) is safer
        if parPF.master == MPI.Comm_rank(comm)
            model_data.logl_YG_history[l,:] = logp_GY(model_data.G, hyper, parPF, Yc)
        end

    end #end of loop over changepoints

    #return likelihood

    return get_final_stats(particle_history, logl_YG_history, model_data.likelihood) #(G_mean, G_mode)

    MPI.Barrier(comm)
    MPI.Finalize() #?

end



function get_final_stats(particle_history, logl_YG_history, likelihood)

    N = size(logl_YG_history)[2]

    num_graphs = size(logl_YG_history)[1]

    p = size(particle_history)[4]

    trajectory_weights = sum(logl_YG_history, dims = 1) #should be 1xN

    normalize_weights(trajectory_weights)

    G_mean = zeros(num_graphs,p,p)

    weighted_G = zeros(N, p, p)

    for i in 1:num_graphs
        weighted_G = (@view trajectory_weights[:]) .* particle_history[i,:,:,:]
        cur_mean = sum(weighted_G[j,:,:] for j in 1:N)
        G_mean[i,:,:] = copy(cur_mean)
    end

    mode = argmax(trajectory_weights)
    G_mode = particle_history[:,mode,:,:]

    return (G_mean, G_mode, likelihood)

end

function copy_states(particles, buffer, resampling_indices, rank, N_per_rank)

    # particle indices this rank has now
    particle_idx_cur =  (rank * N_per_rank) + 1:((rank + 1) * N_per_rank)

    # allocate a fair share of resampled indices to this rank - these are the indices we want
    particle_idx_want = resampling_indices[particle_idx_cur]

    # the ranks that currently have the particles this rank wants
    rank_senders = floor.(Int, (particle_idx_want .- 1) / N_per_rank)

    reqs = Vector{MPI.Request}(undef, 0)

    # sending particles to the processes that want them
    for (i, idx) in enumerate(resampling_indices)

        rank_receiver = floor(Int, (i - 1) / N_per_rank)

        if idx in particle_idx_cur && rank_receiver != rank # checking if the sender has this particle and sender != receiver

            sender_idx = idx - rank * N_per_rank

            req = MPI.Isend(@view(particles[sender_idx,:,:]), rank_receiver, idx, MPI.COMM_WORLD)

            push!(reqs, req)

        end

    end

    #PROBLEM: BitArray not supported by MPI.MPIPtr so we need to covert the particles to another data type

    # receiving particles into a buffer to avoid overwriting while the particles are swapped around
    for (j, sender) in enumerate(rank_senders)

        idx = particle_idx_want[j]

        if sender == rank #if we already have this particle in our rank put it in buffer

            sender_idx = idx - my_rank * N_per_rank
            @view(buffer[j,:,:]) .= @view(particles[sender_idx,:,:])

        else

            req = MPI.Irecv!(@view(buffer[j,:,:]), sender, idx, MPI.COMM_WORLD)
            push!(reqs,req)

        end

    end

    MPI.Waitall!(reqs)

    # ?
    particles = buffer

end

#reserts weights
function reset_weights(weights)
    N = size(weights)[1]
    weights = zeros(N)
end




# modifies resampling_indices in place without returning anything
# TODO:
#systematic resampling
#input: normalised weights (not log) - all weights i.e. master rank
#output: ancestor indices (all ancestor indices)
function resample(resampling_indices, weights)

    #THIS FUNCTION IS SUPPOSED TO BE CALLED BY THE MASTER RANK

    resampling_indices = systematic_resampling(weights)

    #now master rank's resampling_indices are modified so we need to send them to other ranks


end

#input: normalised wights
function systematic_resampling(weights)

    N = length(weights)
    offset = rand() / N
    positions = offset .+ collect(0:1:N-1)/N

    indices = zeros(N)
    cum_sum = cumsum(weights)
    i, j = 1, 1

    while i <= N
        if positions[i] < cum_sum[j]
            indices[i] = j
            i += 1
        else
            j += 1
        end
    end
    return(indices)
end

#input: log weights
# 1. normalizes weights by modifying them in place
# 2. returns the sum of shifted weights (needed to increment likelihood)
function normalize_weights_and_get_sum_max(weights)

    max_logw = maximum(weights)
    weights .-= max_logw
    @. weights = exp(weights)
    sum_shifted_weights = sum(weights)
    weights ./= sum_shifted_weights

    return(sum_shifted_weights, max_logw)

end

function normalize_weights(weights)

    weights .-= maximum(weights)
    @. weights = exp(weights)
    weights ./= sum(weights)

end

mutable struct model
    G :: BitArray{3}
    Gp :: BitArray{3}
    particle_history :: Array{Float64,4}
    logl_YG_history :: Array{Float64,2}
    logl_GG :: Array{Float64,1}
    logl_GY :: Array{Float64,1}
    likelihood :: Float64
end

#initialises the model: G, Gp, particle_history, result/estimate (basically the intended output of the PF)
function init_model(hyper, parPF, N_per_rank, k, rank)

    if MPI.Comm_rank(MPI.COMM_WORLD) == parPF.master
        G = falses(parPF.N, hyper.p, hyper.p)
        Gp = falses(parPF.N, hyper.p, hyper.p)
        logl_GG = zeros(parPF.N)
        logl_GY = zeros(parPF.N)
    else
        G = falses(N_per_rank, hyper.p, hyper.p)
        Gp = falses(N_per_rank, hyper.p, hyper.p)
        logl_GG = zeros(N_per_rank)
        logl_GY = zeros(N_per_rank)
    end

    #everyone stores it
    particle_history = zeros(k+1,parPF.N, hyper.p, hyper.p)
    logl_YG_history = zeros(k+1,parPF.N)

    likelihood = 0.0

    this_model = model(G, Gp, particle_history, logl_YG_history, logl_GG, logl_GY, likelihood)

    return this_model

end

mutable struct model
    G :: BitArray{3}
    Gp :: BitArray{3}
    particle_history :: Array{Float64,4}
    logl_YG_history :: Array{Float64,2}
    logl_GG :: Array{Float64,1}
    logl_GY :: Array{Float64,1}
    likelihood :: Float64
end

#QUESTION: how does master rank work

mutable struct filter_data
    weights :: Array{Float64,1}
    #w_std :: Array{Float64,1}
    resampling_indices :: Array{Int64,1}
    copy_buffer :: BitArray{3}
end


# Initialize arrays used by the filter: weights (logw), w_std, resampling indices
function init_filter(hyper, par_PF, N_per_rank, rank)

    if rank == parPF.master
        weights = zeros(parPF.N) #log weights
        #w_std = zeros(parPF.N)
    else
        weights = zeros(N_per_rank) #log weights
        #w_std = zeros(N_per_rank)
    end

    # ? declared globally

    # every rank has N resampling indices, for correctness they should be the same
    resampling_indices = Array{Int64, 1}(undef, parPF.N) #Array{Int64,1}

    # OPTIONAL: add state variables etc

    # ? dimensions, memory buffer used during copy of the states
    copy_buffer = falses(N_per_rank, hyper.p, hyper.p)

    return filter_data(weights, resampling_indices, copy_buffer)

end
