#=
    INCOMPLETE CHOLESKY FACTORIZATION FOR SVM
    =========================================

    José Manuel Proudinat Silva
    jmps2812@gmail.com
    2016

=#


function kernel_ichol(data::SVM_train_data,
                      kernel::SVM_kernel,
                      tol::Float64)
    #=
    Serial method for Incomplete Cholesky Factorization
    for a SVM Kernel Matrix.
    =#

    n, m = size(data.X)
    H = spzeros(n, m)
    v = [K(data.X[i, :], data.X[i, :], kernel) for i in 1:n]
    ~, pivot = findmax(v)
    k = 1
    I = [pivot]
    J = setdiff(collect(1:n), I)
    norm_v = norm(v)
    print("iter: ", k, "; norm(v): ", norm_v, "\n")
    tol = (1 + norm_v) * tol

    while norm_v > tol

        H[pivot, k] = sqrt(v[pivot])

        for j in J
            Q = data.Y[j] * data.Y[pivot]
            Q *= K(data.X[j,:], data.X[pivot,:], kernel)
            try # k=1
                Q -= sum([H[j, l] * H[pivot, l] for l in 1:(k-1)])
            end
            Q /= H[pivot, k]
            H[j, k] = Q
        end

        v -= H[:, k].^2
        norm_v = norm(v)
        ~, pivot = findmax(v)
        J = setdiff(J, [pivot])
        I = union(I, [pivot])
        k += 1
        print("iter: ", k, "; norm(v): ", norm_v, "\n")

    end

    H = H[:, 1:(k-1)]
    return H

end


@everywhere function kernel_diag(X::Array{Float64, 2}, kernel::SVM_kernel)
    n, ~ = size(X)
    v = [K(X[i, :], X[i, :], kernel) for i=1:n]
    return v
end


@everywhere function ichol_column(X::Array{Float64,2},
                                  X_pivot::Array{Float64,1},
                                  Y::Array{Float64,2},
                                  Y_pivot::Float64,
                                  H::SparseMatrixCSC{Float64,Int64},
                                  H_pivot::SparseMatrixCSC{Float64,Int64},
                                  I::Array{Int64,1},
                                  kernel::SVM_kernel,
                                  k::Int64)

    n, ~ = size(X)
    T = spzeros(n,1)

    for j in 1:n
        if !(j in I)
            Q = Y[j] * Y_pivot
            Q *= K(X[j,:], X_pivot', kernel)
            try
                Q -= sum([H[j, l] * H_pivot[l] for l in 1:(k-1)])
                Q /= H_pivot[k]
            end
            T[j] = Q
        end
    end

    return T

end


function convertsub(x::SubArray{Float64,2,DistributedArrays.DArray{Float64,2,Array{Float64,2}},Tuple{UnitRange{Int64},Colon},0})
    n = length(x)
    return [x[i] for i=1:n]
end


function distributed_kernel_ichol(X::Array{Float64},
                                  Y::Array{Float64},
                                  kernel::SVM_kernel,
                                  tol::Float64)

    #=
    Distributed method for Incomplete Cholesky Factorization
    for a SVM Kernel Matrix.
    =#

    n, m = size(X)
    X = distribute(X)
    Y = distribute(Y)
    
    indexes = X.indexes
    pids = X.pids
    N = length(pids)
    I = [[0] for i in 1:N]
    
    H = spzeros(n, m)
    k = 1

    # Initial diagonal Kernel matrix
    v = [(@spawnat pids[i] kernel_diag(localpart(X), kernel)) for i in 1:N]
    v = reduce(vcat, pmap(fetch, v))
    v = distribute(v)

    # Find pivot
    pivot = pmap(fetch, [(@spawnat pids[i] findmax(localpart(v))) for i in 1:N])
    (pivot, local_pivot_index), pivot_proc_index = findmax(pivot)
    global_pivot_index = indexes[pivot_proc_index][1][local_pivot_index]
    print("pivot: ", pivot, ' ', "index: ", global_pivot_index, "\n")

    # Add pivot to local indexes
    I[pivot_proc_index] = vcat(I[pivot_proc_index], [local_pivot_index])
    X_pivot = convertsub(X[global_pivot_index, :])
    Y_pivot = Y[global_pivot_index]
    H_pivot = H[global_pivot_index, 1:(k-1)]

    while pivot > tol
    
        # Set column pivot value
        H[global_pivot_index, k] = sqrt(pivot)

        T = [
            (@spawnat pids[i] ichol_column(
                localpart(X),
                X_pivot,
                localpart(Y),
                Y_pivot,
                H[indexes[i][1], 1:(k-1)],
                H_pivot,
                I[i],
                kernel,
                k    
                )
            ) for i in 1:N
        ]
        T = reduce(vcat, pmap(fetch, T))
        print(T, "\n")
        H[:, k] += T

        # Update values
        # ...
        v = [(@spawnat pids[i] +(localpart(v), H[indexes[i][1], k])) for i in 1:N]
        v = reduce(vcat, pmap(fetch, v))
        v = distribute(v)
        k += 1

        # Find pivot
        pivot = pmap(fetch, [(@spawnat pids[i] findmax(localpart(v))) for i in 1:N])
        (pivot, local_pivot_index), pivot_proc_index = findmax(pivot)
        global_pivot_index = indexes[pivot_proc_index][1][local_pivot_index]
        print("pivot: ", pivot, ' ', "index: ", global_pivot_index, "\n")

        # Add pivot to local indexes
        I[pivot_proc_index] = vcat(I[pivot_proc_index], [local_pivot_index])
        X_pivot = convertsub(X[global_pivot_index, :])
        Y_pivot = Y[global_pivot_index]
        H_pivot = H[global_pivot_index, 1:(k-1)]

    end

    return H[:, 1:(k-1)]

end
