using LinearAlgebra
using Statistics
using Distributions
using SpecialFunctions

function normalising_constant_mc(G,delta,D,num_it)

    #this works assuming that G is A i.e. a graph which has 0s and 1s only for (i,j) where i < j and (i,j) is an edge (upper triangular)

    #G is as abive but for a prime component Pj of the graph
    #delta is the delta from I(delta, D)
    # D is D[Pj] subgraph of D where Pj is the prime component

    p = size(D, 1)

    T = cholesky(inv(D)) #gives the inverse of D
    diagT = Diagonal(T.U)

    #array of cartesian indices of edges (i,j) where i < j
    E = findall(x->x==1,UpperTriangular(G))

    #array of cartesian indices of non-edges i.e. 0s (i,j) where i < j
    Vbar = findall(x->x==0,UpperTriangular(G) + LowerTriangular(ones(p,p)))

    Vbar = sort([[v[1], v[2]] for v in Vbar])

    Vbar = [CartesianIndex(v[1], v[2]) for v in Vbar]

    #p array – number of 1s in the ith column of A
    k = sum(UpperTriangular(G), dims=1)

    #p array – number of 1s in the ith row of A
    v = transpose(sum(UpperTriangular(G), dims=2))

    T = T.U ./transpose(repeat([T.U[i,i] for i in 1:p], outer=[1,p]))

    idx = I(p)

    Jhat = 0

    C = zeros(p,p)

    B = zeros(p,p)

    for n in 1:num_it

        C = zeros(p,p)

        #edges (i,j) i < j
        C[E] = randn(size(E,1),1)

        #nodes (i,i)
        C[idx] = [sqrt(rand(Chisq(x))) for x in delta.+v]

        #non-edges (i,j) i < j

        #works

        for v in Vbar
            i = v[1]
            j = v[2]

            # (i,j)th entry of B is the sum for free element (i,j)
            B = C*T

            C[i,j] = -B[i,j]

            if (i > 1)
                C[i,j] = C[i,j] - sum( (C[1:(i-1),i] + B[1:(i-1),i]).*(C[1:(i-1),j] + B[1:(i-1),j]) ) ./ C[i,i]
            end
            if isinf(C[i,j])
                break
            end
        end

        Jhat = Jhat + exp(-0.5*sum((C[Vbar]).^2))/num_it

    end

    b = v.+k.+1
    dv2 = (delta.+v)./2
    logC = sum(v./2 .* log(2 .* pi) .+ dv2 .* log(2) .+ [loggamma(d) for d in dv2] .+ (delta .+ b .- 1) .* transpose([log(diagT[i,i]) for i in 1:p]))

    logI = log(Jhat) + logC

    return(logI)

end
