using LightGraphs
using LinearAlgebra
using Graphs
using Statistics

#would be better to define a graph as a class

function matrix_to_graph(A)

    p = size(A, 1)

    g = simple_graph(p, is_directed = false)

    ones = findall(!iszero, A)

    for idx in ones
        (v_in, v_out) = Tuple(idx)
         add_edge!(g, v_in, v_out)
    end

    return g

end

function matrix_to_dict(A)

    dict = Dict()

    p = size(A, 1)

    ones = findall(!iszero, A)

    for i in 1:p

        ls = zeros(Int64, 0)

        col_ones = findall(!iszero, A[:,i])
        row_ones = findall(!iszero, A[i,:])

        for idx in col_ones append!(ls, idx) end

        for idx in row_ones append!(ls, idx) end

        dict[i] = ls

    end

    return dict

end

function gwishart_mode(G_dict, delta, D, N = 100)

    #G_dict is a dict with vertex as key and a list of neighbours as value

    n = length(G_dict)

    L = D ./ (delta - 2.0)

    K = deepcopy(L)

    K_ = deepcopy(K)

    index_rgwish = [filter!(e->e≠j, Vector(range(1, stop = n))) for j in range(1, stop = n)]

    while true

        K_ = deepcopy(K)

        for j in 1:n
            β_hat_j = zeros(n)
            N_j = G_dict[j] #neighbours of vertex j

            # Step 2a and 2b
            if length(N_j) > 0
                β_hat_j[N_j] = K[N_j, :][:, N_j] \ L[N_j, j] #solves the system K[N_j, :][:, N_j] * x = L[N_j, j]
            end

            # Step 2c
            tmp = K[index_rgwish[j], :][:, index_rgwish[j]] * β_hat_j[index_rgwish[j]]
            K[j, index_rgwish[j]] = tmp
            K[index_rgwish[j], j] = tmp
        end

        # Step 3: Stop if converged.
       if Statistics.mean((K .- K_).^2) < 1e-8
           break
       end

    end

    return inv(K) # Step 4

end

# G_V is ...
function hessian(K, G_V, delta)

    n_e = length(G_V)

    H = zeros(n_e, n_e)

    K_inv = inv(K)

    for a in 1: n_e

        (i, j) = G_V[a] #check

        for b in a: n_e

            (l, m) = G_V[b] #check

            if i == j

                if l == m

                    H[a,b] = (K_inv[i,l])^2

                else

                    H[a, b] = 2.0 * K_inv[i, l] * K_inv[i, m]

                end

            else

                if l == m

                    H[a, b] = 2.0 * K_inv[i, l] * K_inv[j, l]

                else

                    H[a, b] = 2.0 * (K_inv[i, l]*K_inv[j, m] + K_inv[i, m]*K_inv[j, l])

                end

            end

        end

    end

    # So far, we have only determined the upper triangle of H.
    H = H + transpose(H) - diagm(diag(H))

    return -0.5 * (delta - 2.0) * H

end

function log_IG_laplace(G, delta, D)


    #Log of Laplace approximation of G-Wishart normalization constant
    #Log of the Laplace approximation of the normalization constant of the G-Wishart
    #distribution outlined by Lenkoski and Dobra (2011, doi:10.1198/jcgs.2010.08181)

    G_dict = matrix_to_dict(G)

    p = length(G_dict)

    K = gwishart_mode(G_dict, delta, D)

    V = []

    # creating duplication matrix
    for (k, val) in G_dict

        push!(V, (k,k))

        for v in val
            if k < v
                push!(V, (k,v))
            end
        end
    end

    h = -0.5 * (tr(transpose(K) * D) - (delta - 2.0)*logabsdet(K)[1])

    H = hessian(K, V, delta)

    # The minus sign in front of `H` is not there in Lenkoski and Dobra (2011, Section 4).
    # I think that it should be there as |H| can be negative while |-H| cannot.

    return h + 0.5*length(V)*log(2.0 * pi) - 0.5*logabsdet(-H)[1]

end
