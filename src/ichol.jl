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
