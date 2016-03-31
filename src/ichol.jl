#=
    INCOMPLETE CHOLESKY FACTORIZATION FOR SVM
    =========================================

    José Manuel Proudinat Silva
    jmps2812@gmail.com
    2016

=#

function kernel_ichol(X::Array{Float64,2},
                      Y::Array{Float64,2},
                      kernel::SVM_kernel,
                      tol::Float64,
                      maxdim::Int64)
    #=

    Description:
    ------------
    Serial method for Incomplete Cholesky Factorization
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

    n, ~ = size(X)
    H = spzeros(n, maxdim)
    v = [K(X[i, :], X[i, :], kernel) for i in 1:n]

    trace = sum(v)
    ~, pivot = findmax(v)

    k = 1
    I = [pivot]
    J = setdiff(collect(1:n), I)
    base_trace = trace
    tol *= 1 + base_trace

    @printf "  iter        trace  \n"
    @printf " ---------------------- \n"
    @printf " %3i      %1.6e  \n" (k-1) trace

    while trace > tol && k <= maxdim

        H[pivot, k] = sqrt(v[pivot])

        for j in J
            Q = Y[j] * Y[pivot]
            Q *= K(X[j,:], X[pivot,:], kernel)
            try # k=1
                Q -= sum([H[j, l] * H[pivot, l] for l in 1:(k-1)])
            end
            Q /= H[pivot, k]
            H[j, k] = Q
        end

        v -= H[:, k].^2

        ~, pivot = findmax(v)
        trace = sum(v)

        J = setdiff(J, [pivot])
        I = union(I, [pivot])
        k += 1

        @printf " %3i      %1.6e  \n" (k-1) trace

    end

    H = H[:, 1:(k-1)]
    return H

end


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


@everywhere function ichol_column(X::Array{Float64,2},
                                  X_pivot::Array{Float64,1},
                                  Y::Array{Float64,2},
                                  Y_pivot::Float64,
                                  H::SparseMatrixCSC{Float64,Int64},
                                  H_pivot::SparseMatrixCSC{Float64,Int64},
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
    T = spzeros(n,1)

    for j in 1:n
        if !(j in I)
            Q = Y[j] * Y_pivot
            Q *= K(X[j,:], X_pivot', kernel)
            try
                Q -= sum([H[j, l] * H_pivot[l] for l in 1:(k-1)])
            end
            Q /= H_pivot[k]
            T[j] = Q
        end
    end

    return T

end


function convertsub(x::SubArray{Float64,2,DistributedArrays.DArray{Float64,2,Array{Float64,2}},Tuple{UnitRange{Int64},Colon},0})
    #=

    Description:
    ------------
    Convert a SubArray{Float64} subfactor to an Array{Float64}.

    Input:
    ------
        - x : subarray to convert.

    Output:
    -------
        - Array{Float64,1} | Array converted.

    =#

    n = length(x)
    return [x[i] for i=1:n]
end


@everywhere function update_diag(v::Array{Float64},
                                 H::SparseMatrixCSC{Float64,Int64})
    #=

    Description:
    ------------
    Auxiliar function to update the diagonal Kernel matrix residuals.

    Input:
    ------
        - v : Kernel matrix residuals.
        - H : Kernel matrix factor values.

    Output:
    -------
        - Array{Float64} | updated Kernel matrix residuals.

    =#

    return v - (H.^2)
end


function parallel_kernel_diag(X::DistributedArrays.DArray,
                              kernel::SVM_kernel,
                              v::DistributedArrays.DArray)
    #=

    Description:
    ------------
    Compute kernel matrix diagonal for DistributedArrays.

    Input:
    ------
        - X : train data characteristics.
        - v : kernel diagonal values

    Output:
    -------
        - nothing

    =#

    refs = [(@spawnat w kernel_diag(localpart(X), kernel, localpart(v))) for w in procs(X)]
end


function distributed_kernel_ichol(X::Array{Float64},
                                  Y::Array{Float64},
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

    H = spzeros(n, maxdim)
    k = 1

    # Initial diagonal Kernel matrix
    v = [(@spawnat pids[i] kernel_diag(localpart(X), kernel)) for i in 1:N]
    v = reduce(vcat, pmap(fetch, v))
    ttt = toq()
    print("Diagonal kernel: ", ttt, "\n")

    tic()
    trace = sum(v)
    ttt = toq()
    print("Trace: ", ttt, "\n")

    tic()
    v = distribute(v)

    # Find pivot
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
        H[global_pivot_index, k] = sqrt(pivot)
        I[pivot_proc_index] = vcat(I[pivot_proc_index], [local_pivot_index])
        X_pivot = convertsub(X[global_pivot_index, :])
        Y_pivot = Y[global_pivot_index]
        H_pivot = H[global_pivot_index, :]
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
                H[indexes[i][1], 1:(k-1)],
                H_pivot,
                I[i],
                kernel,
                k
                )
            ) for i in 1:N
        ]
        T = reduce(vcat, pmap(fetch, T))
        H[:, k] += T
        ttt = toq()
        print(k, " | Compute column: ", ttt, "\n")

        # Update values
        # ...
        tic()
        v = [(@spawnat pids[i] update_diag(localpart(v), H[indexes[i][1], k])) for i in 1:N]
        v = reduce(vcat, pmap(fetch, v))
        ttt = toq()
        print(k, " | Update diagonal: ", ttt, "\n")

        tic()
        trace = sum(v)
        ttt = toq()
        print(k, " | Trace: ", ttt, "\n")

        tic()
        v = distribute(v)
        k += 1

        # Find pivot
        pivot = pmap(fetch, [(@spawnat pids[i] findmax(localpart(v))) for i in 1:N])
        (pivot, local_pivot_index), pivot_proc_index = findmax(pivot)
        global_pivot_index = indexes[pivot_proc_index][1][local_pivot_index]
        ttt = toq()
        print(k, " | Find pivot: ", ttt, "\n")
        print("trace: ", trace, ' ', "index: ", global_pivot_index, "\n")

    end

    tic()
    return H[:, 1:(k-1)]
    ttt = toq()
    print("Get final matrix: ", ttt, "\n")

end
