
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
        - Float64 | local trace value.

    =#

    n, ~ = size(X)
    v[1:n] = [K(X[i, :], X[i, :], kernel) for i=1:n]
    return sum(v), findmax(v)

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
    X[i, j] = v
end


@everywhere function pivot_data(X::Array{Float64,2},
                                Y::Array{Float64,2},
                                H::Array{Float64,2},
                                i::Int64,
                                j::Int64,
                                v::Float64)
    #=

    Description:
    ------------
    Set H[i,j] = v. And return pivot data.
    (X[i,:], Y[i], H[i,:])

    =#

    H[i, j] = v
    return X[i, :], Y[i], H[i, :]

end


@everywhere function ichol_column(X::Array{Float64,2},
                                  X_pivot::Array{Float64,2},
                                  Y::Array{Float64,2},
                                  Y_pivot::Float64,
                                  H::Array{Float64,2},
                                  H_pivot::Array{Float64,2},
                                  v::Array{Float64,2},
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
        - v : kerel matrix diagonal residuals.
        - I : indexes to compute.
        - kernel : kernel type.
        - k : iteration value, column index.

    Output:
    -------
        - sum(v) : Float64 | local trace value.
        - findmax(v) : Tuple{Float64,Int64} | local pivot value.

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

    v[1:n] = v - H[:, k].^2

    return sum(v), findmax(v)

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

    n, ~ = size(X)
    X = distribute(X)
    Y = distribute(Y)

    indexes = X.indexes
    pids = X.pids
    N = length(pids)
    I = [[0] for i in 1:N]

    v = dzeros(n, 1)
    H = dzeros(n, maxdim)
    k = 1

    # Initial diagonal Kernel matrix
    refs = [
        @spawnat pids[i] kernel_diag(localpart(X), kernel, localpart(v))
        for i in 1:N
    ]
    calls = pmap(fetch, refs)
    trace_list = [calls[i][1] for i = 1:N]
    pivot_list = [calls[i][2] for i = 1:N]
    trace = sum(trace_list)
    tol *= 1 + trace

    # Find pivot
    (pivot, local_pivot_index), pivot_proc_index = findmax(pivot_list)
    global_pivot_index = indexes[pivot_proc_index][1][local_pivot_index]

    @printf "  iter        trace  \n"
    @printf " ---------------------- \n"
    @printf " %3i      %1.6e  \n" (k-1) trace

    while trace > tol && k <= maxdim

        # Add pivot to local indexes
        refs = @spawnat pids[pivot_proc_index] pivot_data(
          localpart(X),
          localpart(Y),
          localpart(H),
          local_pivot_index,
          k,
          sqrt(pivot)
        )
        X_pivot, Y_pivot, H_pivot = fetch(refs)
        I[pivot_proc_index] = vcat(I[pivot_proc_index], [local_pivot_index])

        # Set column pivot value
        refs = [
            (@spawnat pids[i] ichol_column(
                localpart(X),
                X_pivot,
                localpart(Y),
                Y_pivot,
                localpart(H),
                H_pivot,
                localpart(v),
                I[i],
                kernel,
                k
                )
            ) for i in 1:N
        ]

        calls = pmap(fetch, refs)
        trace_list = [calls[i][1] for i = 1:N]
        pivot_list = [calls[i][2] for i = 1:N]
        trace = sum(trace_list)
        k += 1

        (pivot, local_pivot_index), pivot_proc_index = findmax(pivot_list)
        global_pivot_index = indexes[pivot_proc_index][1][local_pivot_index]

        @printf " %3i      %1.6e  \n" (k-1) trace

    end

    H = convert(Array, H)
    return sparse(H[:, 1:(k-1)])

end
