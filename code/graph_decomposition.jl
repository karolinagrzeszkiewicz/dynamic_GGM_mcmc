module graph_decomposition
using StatsBase
using Logging
using LinearAlgebra

"""
    edges_to_adjacency(edges, num_vertices)

Constructs an adjacency matrix from the edges of a graph.

# Examples
```julia-repl
julia> edges_to_adjacency(6, ((1,2),(2,3),(2,4),(2,5),(3,5),(4,5),(5,6)))
6×6 Array{Int8,2}:
 0  1  0  0  0  0
 1  0  1  1  1  0
 0  1  0  0  1  0
 0  1  0  0  1  0
 0  1  1  1  0  0
 0  0  0  0  1  0
```
"""
function edges_to_adjacency(num_vertices::Int64, edges::NTuple{N, Tuple{Int64,Int64}}) where N
    adj = zeros(Int64, num_vertices, num_vertices)
    for e in edges
        adj[e[1], e[2]] = 1
        adj[e[2], e[1]] = 1
    end # for
    return adj
end # edges_to_adjacency

"""
    find_neighbours(adjacency_matrix)

Find the neighbours of a vertex given the adjacency matrix.
"""
function find_neighbours(adjacency_matrix::Array{Int64,2})
    neighbours = Dict{Int64, Array{Int64,1}}()
    for i = 1:size(adjacency_matrix, 2)
        neighbours[i] = findall(x -> x == 1, adjacency_matrix[:, i])
    end
    return neighbours
end # find_neighbours

"""
    is_graph_complete(adjacency_matrix, sub_graph_vertices)

Given the adjacency matrix and the sub-graph vertices, check whether the sub-graph is complete.
"""
function is_sub_graph_complete(adjacency_matrix::Array{Int64,2}, sub_graph_vertices::Array{Int64})
    sub_mat = adjacency_matrix[sub_graph_vertices, sub_graph_vertices]
    n_edges = sum(sub_mat) / 2
    return n_edges == binomial(size(sub_graph_vertices, 1), 2)
end # is_sub_graph_complete

"""
    max_cardinality_search(adjacency_matrix)

Perform a maximum cardinality search.
"""
function max_cardinality_search(adjacency_matrix::Array{Int64,2}) where N
    # Initialise
    num_vertices = size(adjacency_matrix, 1)
    numbering = zeros(Int64, 1, num_vertices)
    vertex_ranking = zeros(Int64, 1, num_vertices)
    neighbours = find_neighbours(adjacency_matrix)
    ladder_nodes = Dict{Int64, Array{Int64,1}}()

    # Do for the first iteration
    init_vertex = rand(1 : num_vertices)
    numbering[1] = init_vertex
    remaining_vertices = filter!(x -> x ≠ init_vertex, collect(1 : num_vertices))
    vertex_ranking[neighbours[init_vertex]] .+= 1
    pi_old = intersect(numbering, neighbours[init_vertex])

    # Iterate over remaining vertices
    count = 2
    while count <= num_vertices
        remaining_vertex_ranking = vertex_ranking[remaining_vertices]
        ix = findall(x->x == maximum(remaining_vertex_ranking), remaining_vertex_ranking)
        next_vertex = rand(remaining_vertices[ix])

        # Check for chordal graph
        pi_new = intersect(numbering, neighbours[next_vertex])
        if !is_sub_graph_complete(adjacency_matrix, pi_new)
            return false, numbering, ladder_nodes
        end # if

        # Prime components
        if size(pi_new, 1) < size(pi_old, 1) + 1
            ladder_nodes[numbering[count - 1]] = union(numbering[count - 1], pi_old)
        end # if

        vertex_ranking[intersect(remaining_vertices, neighbours[next_vertex])] .+= 1
        numbering[count] = next_vertex
        remaining_vertices = filter!(x -> x ≠ next_vertex, remaining_vertices)

        pi_old = pi_new
        count += 1
    end # while
    ladder_nodes[numbering[count - 1]] = union(numbering[count - 1], pi_old)
    cliques = [sort(nodes) for nodes in values(ladder_nodes)]

    return true, numbering, ladder_nodes, cliques # is_chordal, numbering, ladder_nodes, cliques
end # max_cardinality_search

"""
    running_intersection(components)

Find the running intersection of the prime components.
"""
function running_intersection(components)

    #### Method 1
    #sep_sets_1 = []
    #for i in 1:length(components)
    #    c = components[i]
    #    c_rem = components[i+1:end]
    #    card = [length(intersect(c, c_r)) for c_r in c_rem]
    #    if length(card) > 0
    #        ix = findall(x -> x == maximum(card), card)[1]
    #        sep = intersect(c, c_rem[ix])
    #        if length(sep) > 0
    #            push!(sep_sets_1, sep)
    #        end
    #    end
    #end
    #
    #cum_union = components[1]
    #sep_sets_2 = []
    #for c in components[2:end]
    #    sep = intersect(c, cum_union)
    #    if length(sep) > 0
    #        push!(sep_sets_2, sep)
    #    end
    #  cum_union = union(cum_union, c)
    #end
    #
    #if length(sep_sets_1) >= length(sep_sets_2)
    #    sep_sets = sep_sets_1
    #else
    #    sep_sets = sep_sets_2
    #end
    #
    #if length(intersect(components[1], components[end])) > length(sep_sets[end])
    #    sep_sets[end] = intersect(components[1], components[end])
    #end

    #### Method 2
    #sep_sets = []
    #c_rem = copy(components)
    #for i in 1:length(components)
    #    c = components[i]
    #    c_rem_less_c = [c_r for c_r in c_rem if c_r ≠ c]
    #    card = [length(intersect(c, c_r)) for c_r in c_rem_less_c]
    #    ix = findall(x -> x == maximum(card), card)[1]
    #    push!(sep_sets, intersect(c, c_rem_less_c[ix]))
    #    #c_rem = [c_r for c_r in c_rem if c_r != c_rem_less_c[ix]]
    #end

    #### Method 3
    #sep_sets = []
    #for i in 1:length(components)-1
    #    c = components[i]
    #    c_rem = components[i+1:end]
    #    card = [length(intersect(c, c_r)) for c_r in c_rem]
    #    ix = findall(x -> x == maximum(card), card)[1]
    #    push!(sep_sets, intersect(c, c_rem[ix]))
    #end
    #if length(sep_sets) > 0 && length(sep_sets[end]) == 0
    #     sep_sets[end] = intersect(components[1], components[end])
    #end

    #### Method 4
    #sep_sets = unique([intersect(components[i],components[j]) for i in 1:length(components) for j in i+1:length(components)])
    #sep_sets = [s for s in sep_sets if length(s) > 0]

    #### Method 5
    function helper1(components, start)
        comps = copy(components)
        comps = vcat(comps[start:end], comps[1:start-1])
        sep_sets = []
        cardintality = 0
        for i in 1:length(comps)
            c = comps[i]
            c_rem = comps[i+1:end]
            card = [length(intersect(c, c_r)) for c_r in c_rem]
            if length(card) > 0
                ix = findall(x -> x == maximum(card), card)[1]
                sep = intersect(c, c_rem[ix])
                if length(sep) > 0
                    push!(sep_sets, sep)
                    cardintality += length(sep)
                end
            end
        end
        return sep_sets, cardintality + length(sep_sets)
    end

    function helper2(components, start)
        comps = copy(components)
        comps = vcat(comps[start:end], comps[1:start-1])
        cum_union = comps[1]
        sep_sets = []
        card = 0
        for c in comps[2:end]
            sep = intersect(c, cum_union)
            if length(sep) > 0
                push!(sep_sets, sep)
                card += length(sep)
            end
          cum_union = union(cum_union, c)
        end
        return sep_sets, card + length(sep_sets)
    end

    separator_sets = []
    max_card = 0
    for i in 1:length(components)
        tmp1 = helper1(components, i)
        tmp2 = helper2(components, i)
        if tmp1[2] >= tmp2[2] && tmp1[2] > max_card
            separator_sets = tmp1[1]
            max_card = tmp1[2]
        elseif tmp2[2] > tmp1[2] && tmp2[2] > max_card
            separator_sets = tmp2[1]
            max_card = tmp2[2]
        end
    end

    return separator_sets
end # running_intersection

"""
    min_deficiency_search(adjacency_matrix)

Perform a minimum deficiency search (triangulation).
"""
function min_deficiency_search(adjacency_matrix::Array{Int64,2}) where N
    function opt_criterion(adjacency_matrix, vertex, neighbours)
        index = sort(union(vertex, neighbours))
        new_edges = ones(Int64, length(index), length(index)) - adjacency_matrix[index,index] - Matrix{Int64}(I, length(index), length(index))
        return sum(new_edges) / 2
    end

    num_vertices = size(adjacency_matrix, 1)
    new_adjacency_matrix = copy(adjacency_matrix)
    remaining_vertices = collect(1 : num_vertices)

    while length(remaining_vertices) > 0
        neighbours = find_neighbours(new_adjacency_matrix)
        remaining_vertex_ranking = [opt_criterion(new_adjacency_matrix, v, intersect(neighbours[v], remaining_vertices)) for v in remaining_vertices]
        ix = findall(x->x == minimum(remaining_vertex_ranking), remaining_vertex_ranking)
        next_vertex = rand(remaining_vertices[ix])

        ne = intersect(neighbours[next_vertex], remaining_vertices)
        tmp = sort(union(next_vertex, ne))
        new_adjacency_matrix[tmp,tmp] = ones(Int64, length(tmp), length(tmp))
        new_adjacency_matrix[diagind(new_adjacency_matrix)] = zeros(Int64,size(new_adjacency_matrix)[1])

        remaining_vertices = filter!(x -> x ≠ next_vertex, remaining_vertices)
    end
    new_edges = new_adjacency_matrix - adjacency_matrix
    return new_adjacency_matrix, new_edges
end # min_deficiency_search

"""
    decompose_graph(adjacency_matrix)

Decompose graph into prime components and separators.
"""
function decompose_graph(adjacency_matrix::Array{Int64,2}) where N
    p = size(adjacency_matrix)[1]
    symmetric_adjacency_matrix = Array(Symmetric(adjacency_matrix))

    cardinality_search = max_cardinality_search(symmetric_adjacency_matrix)
    is_chordal = cardinality_search[1]
    if cardinality_search[1]
        complete_components = cardinality_search[end]
        incomplete_components = []
        separators = running_intersection(sort(complete_components))
    else
        new_adjacency_matrix, new_edges = min_deficiency_search(symmetric_adjacency_matrix)
        cardinality_search = max_cardinality_search(new_adjacency_matrix)
        new_edges = findall(x -> x == 1, UpperTriangular(new_edges))
        incomplete_components = [sort(reduce((y, z) -> union(y, z), filter!(x -> length(intersect([e[1],e[2]],x)) == 2, copy(cardinality_search[end])))) for e in new_edges]
        complete_components = []
        for c in cardinality_search[end]
            if all([length(intersect(c,i)) ≠ length(c) for i in incomplete_components])
               push!(complete_components, c)
            end
        end
        #complete_components = unique([c for c in cardinality_search[end] for i in incomplete_components if length(intersect(c,i)) ≠ length(c)])
        components = []
        append!(components, complete_components)
        append!(components, incomplete_components)
        separators = running_intersection(sort(components))
        #separators = running_intersection(cardinality_search[end])
        #separators = [filter!(x -> length(intersect([e[1],e[2]],x)) < 2, separators) for e in new_edges][1]
        #separators = [s for s in separators if length(s)>0]
    end
    return is_chordal, complete_components, incomplete_components, separators
end # decompose_graph

end # module
