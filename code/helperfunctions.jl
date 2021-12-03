#HELPER FUNCTIONS

function delete_changepoint(S, s)

    Spr = []

    for i in S
        if i != s
            push!(Spr, i)
        end

    end
    return(Spr)

end

function append_changepoint(S, s)

    if (!isnothing(S)) k = length(S) else k = 0 end

    Spr = []

    i = 1

    s_pushed = false

    if k == 0

        Spr = [s]

    else

        while i < (k+1)
            if (S[i] < s || s_pushed)
                push!(Spr, S[i])
                i += 1
            elseif (S[i] >= s && !s_pushed)
                push!(Spr, s)
                s_pushed = true
            end
        end
        if !s_pushed
            push!(Spr, s)
        end

    end

    return(Spr)
end

#for a given lost of changepoints S returns the list of empty (and valid) spots for a new changepoint

function get_empty_list(S, T, p)

    if (!isnothing(S)) k = length(S) else k = 0 end

    empty = []

    if k == 0
        empty = Array(range(p+3, T -(p+2), step = 1))

    else

        for i in 1:k

            if i == 1

                if S[1] > 2*(p+2)+1
                    values = range(p+3, S[1]-(p+2)-1, step = 1)
                    for v in values
                        push!(empty, v)
                    end
                end

            else

                if S[i]-S[i-1] - 1 > 2*(p+2)
                    values = range(S[i-1]+p+3, S[i]-(p+2)-1, step = 1)
                    for v in values
                        push!(empty, v)
                    end
                end

            end

        end

        if T - S[k] > 2*(p+2)
            values = range(S[k]+p+3, T-(p+2), step = 1)
            for v in values
                push!(empty, v)
            end
        end

    end

    return(empty)

end

#function to find the number of empty spots for a new changepoint

function find_empty(S, T, p)

    k = Int64

    if (!isnothing(S)) k = length(S) else k = 0 end

    if k == 0
        res = T - 2*(p+2)
    else
        res = 0
        for i in 1:k
            if i == 1
                dist = max(0, S[i] - (p+3)-(p+2))
                if S[i] < p+3
                    error("invalid changepoints")
                end
            else
                dist = max(0, (S[i]-(p+2)) - (S[i-1]+(p+2)) - 1)
                if (S[i] - S[i-1]-1) < (p+2)
                    error("invalid changepoints")
                end
            end
            res += dist
        end
        dist = max(0, T - (p+2) - (S[k] + (p+2)))
        res += dist
        if S[k] > T - (p+2)
            error("invalid changepoints")
        end
    end

    return(res)

end

function is_valid(s_pr, S, T, p)

    k = Int64

    if (!isnothing(S)) k = length(S) else k = 0 end

    is_valid = (s_pr > p+2) && (s_pr < T-(p+1))

    if k > 0

        for i in 1:k
            valid = abs(s_pr-S[i])-1 >= (p+2)
            is_valid = is_valid && valid
        end

    end

    return(is_valid)

end

function sum_values_of_keys_smaller_than(dicte, upper)
    res = 0

    for key in keys(dicte)
        if key < upper
            res += dicte[key]
        end
    end

    return(res)

end


function get_permutations(T, p, k)

    if k > 0
        #initialise range and combinations for S_1
        range_s = range(p+2+1, T - (p+2), step = 1)
        combinations = Dict()
        for s in range_s
            combinations[s] = 1
        end
        #the sum of values in combinations is the number of permutations for k = 1

        if k > 1

            for i in range(2, k, step = 1)
                range_i = range(i*(p+2)+1, T - (p+2), step = 1)
                combinations_i = Dict()
                #we assume s_i is the last changepoint and look for the number of possible combinations for s_1:(i-1)
                for s_i in range_i
                    combinations_i[s_i] = sum_values_of_keys_smaller_than(combinations, s_i - (p+2))
                end

                #update combinations, range
                range_s = range_i
                combinations = combinations_i


            end
        end

        res = 0

        for (key, value) in combinations
            res += value
        end

    else

        res = 1

    end

    return(res)

end

function are_valid(S, T, p)

    if (isnothing(S)) k = 0 else k = length(S) end

    if k > 0
        if sort(S) != S
            error("changepoints not sorted")
        end
    end

    valid = true

    if k > 0
         for i in 1:k
            if i == 1
                valid = valid && (S[1] > p+2)
            else
                valid = valid && (S[i]-S[i-1]-1 >= (p+2))
            end
         end
        valid = valid && (S[k] <= T-(p+2))

    end
    return(valid)
end
