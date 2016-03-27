
@everywhere function kernel_diag(X::Array{Float64,2},
                                 kernel::SVM_kernel,
                                 v::Array{Float64,2})
    #=

    Description:
    ------------
    Compute the diagonal of a Kernel matrix.

    Input:
    ------
        - X : train data characteristics.
        - kernel : kernel type.
        - v : kernel diagonal values.

    Output:
    -------
        - nothing

    =#

    n, ~ = size(X)
    v[1:n] = [K(X[i, :], X[i, :], kernel) for i=1:n]

end


@everywhere function get_row(X::Array{Float64,2}, k::Int64)
    #=

    Description:
    ------------
    Return the k'th row from X Array.

    =#
    return X[k, :]
end


@everywhere function get_column(X::Array{Float64,2}, k::Int64)
    #=

    Description:
    ------------
    Return the k'th column from X Array.

    =#
    return X[:, k]
end


@everywhere function set_value(X::Array{Float64,2},
                               i::Int64,
                               j::Int64,
                               v::Float64)
    #=

    Description:
    ------------
    Set X[i,j] = v.

    =#
    X[i,j] = v
end


@everywhere function ichol_column(X::Array{Float64,2},
                                  X_pivot::Array{Float64,2},
                                  Y::Array{Float64,2},
                                  Y_pivot::Float64,
                                  H::Array{Float64,2},
                                  H_pivot::Array{Float64,2},
                                  I::Array{Int64,1},
                                  kernel::SVM_kernel,
                                  k::Int64)
    #=

    Description:
    ------------
    Compute a column from an Incomplete Cholesky factor.

    Input:
    ------
        - X : train data characteristics.
        - X_pivot : column pivot data instance.
        - Y : train data labels.
        - Y_pivot : column pivot label.
        - H : incomplete Cholesky factor previous values.
        - H_pivot : incomplete Cholesky factor pivot value.
        - I : indexes to compute.
        - kernel : kernel type.
        - k : iteration value, column index.

    Output:
    -------
        - T : SparseMatrixCSC{Float64,Int64} | Kernel matrix factor column.

    =#

    n, ~ = size(X)

    for j in 1:n
        if !(j in I)
            Q = Y[j] * Y_pivot
            Q *= K(X[j,:], X_pivot, kernel)
            try
                Q -= sum([H[j, l] * H_pivot[l] for l in 1:(k-1)])
            end
            Q /= H_pivot[k]
            H[j, k] = Q
        end
    end

end


@everywhere function update_diag(v::Array{Float64,2},
                                 H::Array{Float64,2},
                                 k::Int64)
    #=

    Description:
    ------------
    Auxiliar function to update the diagonal Kernel matrix residuals.

    =#

    n = length(v)
    v[1:n] = v - H[:, k].^2
end


function distributed_kernel_ichol(X::Array{Float64,2},
                                  Y::Array{Float64,2},
                                  kernel::SVM_kernel,
                                  tol::Float64,
                                  maxdim::Int64)
    #=

    Description:
    ------------
    Distributed method for Incomplete Cholesky Factorization
    for a SVM Kernel Matrix.

    Input:
    ------
        - X : train data characteristics.
        - Y : train data labels.
        - kernel : kernel type.
        - tol : ichol approximation tolerance.
        - maxdim : max dimension of the approximation matrix.

    Output:
    -------
        - H : SparseMatrixCSC{Float64,Int64} | Kernel matrix factor.

    =#

    tic()
    n, ~ = size(X)
    X = distribute(X)
    Y = distribute(Y)
    ttt = toq()
    print("Distribute data: ", ttt, "\n")

    tic()
    indexes = X.indexes
    print(indexes, "\n\n")
    pids = X.pids
    N = length(pids)
    I = [[0] for i in 1:N]

    v = dzeros(n, 1)
    H = dzeros(n, maxdim)
    k = 1

    # Initial diagonal Kernel matrix
    [@spawnat pids[i] kernel_diag(localpart(X), kernel, localpart(v)) for i in 1:N]
    ttt = toq()
    print("Diagonal kernel: ", ttt, "\n")

    tic()
    trace = sum(v)
    ttt = toq()
    print("Trace: ", ttt, "\n")

    # Find pivot
    tic()
    pivot = pmap(fetch, [(@spawnat pids[i] findmax(localpart(v))) for i in 1:N])
    (pivot, local_pivot_index), pivot_proc_index = findmax(pivot)
    ttt = toq()
    print("Find pivot: ", ttt, "\n")

    tol *= 1 + trace
    global_pivot_index = indexes[pivot_proc_index][1][local_pivot_index]
    print("trace: ", trace, ' ', "index: ", global_pivot_index, "\n")

    while trace > tol && k <= maxdim

        # Add pivot to local indexes
        tic()
        @spawnat pids[pivot_proc_index] set_value(localpart(H), local_pivot_index, k, sqrt(pivot))
        I[pivot_proc_index] = vcat(I[pivot_proc_index], [local_pivot_index])
        X_pivot = fetch(@spawnat pids[pivot_proc_index] get_row(localpart(X), local_pivot_index))
        Y_pivot = Y[global_pivot_index]
        H_pivot = fetch(@spawnat pids[pivot_proc_index] get_row(localpart(H), local_pivot_index))
        ttt = toq();
        print(k, " | Pivot values: ", ttt, "\n")

        # Set column pivot value
        tic()
        T = [
            (@spawnat pids[i] ichol_column(
                localpart(X),
                X_pivot,
                localpart(Y),
                Y_pivot,
                localpart(H),
                H_pivot,
                I[i],
                kernel,
                k
                )
            ) for i in 1:N
        ]
        # print(H[:,k], "\n")
        ttt = toq()
        print(k, " | Compute column: ", ttt, "\n")

        # Update values
        tic()
        [@spawnat pids[i] update_diag(localpart(v), localpart(H), k) for i in 1:N]
        ttt = toq()
        print(k, " | Update diagonal: ", ttt, "\n")

        tic()
        trace = sum(v)
        ttt = toq()
        print("Trace: ", ttt, "\n")
        k += 1

        tic()
        pivot = pmap(fetch, [(@spawnat pids[i] findmax(localpart(v))) for i in 1:N])
        (pivot, local_pivot_index), pivot_proc_index = findmax(pivot)
        global_pivot_index = indexes[pivot_proc_index][1][local_pivot_index]
        ttt = toq()
        print("Find pivot: ", ttt, "\n")
        print("trace: ", trace, ' ', "index: ", global_pivot_index, "\n")

    end

    tic()
    H = convert(Array, H)
    ttt = toq()
    print("Get final matrix: ", ttt, "\n")
    return H[:, 1:(k-1)]

end
