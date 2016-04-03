# VERSION >= v"0.4.0-dev+6641" && __precompile__()
# module ParallelSVM

importall Base
@everywhere using DistributedArrays

function -(x::DArray) return -1 .* x end
(-)(x::Number, A::DArray) = x .- A


@everywhere type SVM_kernel
    kernel::AbstractString
    arg1::Float64
    arg2::Float64
end


@everywhere type SVM_predictor
    kernel::SVM_kernel
    C::Float64
    bias::Float64
    weights::Array{Float64,1}
    support_vectors::Array{Float64,2}
    support_vector_labels::Array{Float64,2}
end


@everywhere function K(x::Array{Float64}, y::Array{Float64}, kernel::SVM_kernel)
    if kernel.kernel == "gaussian"
        return exp(-kernel.arg1 * norm(x - y))
    elseif kernel.kernel == "linear"
        return (x * y')[1]
    else
        return 0.0
    end
end


"""
    compute_bias(predictor::SVM_predictor)

Compute the bias constant of the trained `SVM_predictor`.
"""
function compute_bias(predictor::SVM_predictor)
    predictor.bias = 0
    M = length(predictor.weights)
    bias = 0
    for i = 1:M
        bias += (
            predictor.support_vector_labels[i] -
            predict_value(
                predictor,
                predictor.support_vectors[i, :]
            )
        )
    end
    bias /= M
    predictor.bias = bias
    return predictor
end


"""
    predict_value(predictor::SVM_predictor, x::Array{Float64,2})

Compute the prediction value of a data instance.
i.e.    f(x) = sum( alpha_i * y_i * K(X_i, x_i) )
"""
function predict_value(predictor::SVM_predictor, x::Array{Float64,2})
    result = predictor.bias
    M = length(predictor.weights)
    for i = 1:M
        result += (
            predictor.weights[i] *
            predictor.support_vector_labels[i] *
            K(predictor.support_vectors[i, :], x,
              predictor.kernel)
        )
    end
    return result
end


"""
    predict(predictor::SVM_predictor, x::Array{Float64,2})

Compute the prediction value sign of a data instance.
i.e.    f(x) = sign(sum( alpha_i * y_i * K(X_i, x_i) ))
"""
function predict(predictor::SVM_predictor, x::Array{Float64,2})
    value = predict_value(predictor, x)
    return sign(value) * 1.0
end


"""
    predict_matrix(predictor::SVM_predictor, X::Array{Float64,2})

Compute the prediction value signs of a data matrix.
"""
function predict_matrix(predictor::SVM_predictor, X::Array{Float64,2})
    n, ~ = size(X)
    preds = [predict(predictor, X[i, :]) for i=1:n]
    return preds
end


"""
    kernel_ichol(X::Array{Float64,2},
        Y::Array{Float64,2},
        kernel::SVM_kernel,
        tol::Float64,
        maxdim::Int64)

Serial method for Incomplete Cholesky Factorization for a SVM Kernel Matrix.

`X` : train data characteristics.
`Y` : train data labels.
`kernel` : kernel type.
`tol` : ichol approximation tolerance.
`maxdim` : max dimension of the approximation matrix.
"""
function kernel_ichol(X::Array{Float64,2},
    Y::Array{Float64,2},
    kernel::SVM_kernel,
    tol::Float64,
    maxdim::Int64)

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

    n, ~ = size(X)
    v[1:n] = [K(X[i, :], X[i, :], kernel) for i=1:n]
    return sum(v), findmax(v)
end


@everywhere function pivot_data(X::Array{Float64,2},
    Y::Array{Float64,2},
    H::Array{Float64,2},
    i::Int64,
    j::Int64,
    v::Float64)

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


"""
    distributed_kernel_ichol(X::Array{Float64,2},
        Y::Array{Float64,2},
        kernel::SVM_kernel,
        tol::Float64,
        maxdim::Int64)

Distributed method for Incomplete Cholesky Factorization
for a SVM Kernel Matrix.

`X` : train data characteristics.
`Y` : train data labels.
`kernel` : kernel type.
`tol` : ichol approximation tolerance.
`maxdim` : max dimension of the approximation matrix.
"""
function distributed_kernel_ichol(X::Array{Float64,2},
    Y::Array{Float64,2},
    kernel::SVM_kernel,
    tol::Float64,
    maxdim::Int64)

    n, ~ = size(X)
    X = distribute(X)
    Y = distribute(Y)

    indexes = X.indexes
    pids = X.pids
    N = length(pids)
    I = [[0] for i in 1:N]

    v = dzeros(n, 1)
    H = dzeros((n, maxdim), pids, [N,1])
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


"""
    spdiag(d::Array{Float64})

Creates a sparse diagonal matrix with values in d.
"""
function spdiag(d::Array{Float64})
    n = length(d)
    D = spzeros(n, n)
    for i in 1:n
      D[i, i] = d[i]
    end
    return D
end


"""
    function smw_solve(d::Array{Float64,2},
        V::Array{Float64,2},
        w::Array{Float64,2})

This method solves the problem:
    (D + VV')u = w
where D is a diagonal matrix (diag(d));
using the Sherman-Morrison-Woodburry formula.
"""
function smw_solve(d::Array{Float64,2},
    V::Array{Float64,2},
    w::Array{Float64,2})

    n, m = size(V)
    z = w ./ d
    DV = spdiag(1./d) * V
    A = speye(m) + V' * DV
    b = V' * z
    t = cholfact(A) \ b
    u = z - DV * t
    return u
end


"""
    function smw_solve(d::Array{Float64,2},
        V::Array{Float64,2},
        w::Array{Float64,2})

This method solves the problem:
    (D + VV')u = w
where D is a diagonal matrix (diag(d));
using the Sherman-Morrison-Woodburry formula.
"""
function smw_solve(d::Array{Float64,2},
    V::SparseMatrixCSC{Float64,Int64},
    w::Array{Float64,2})

    n, m = size(V)
    z = w ./ d
    DV = spdiag(1./d) * V
    A = speye(m) + V' * DV
    b = V' * z
    # t = cholfact(A) \ b  # TODO: Corregir errores numéricos
    t = A \ b
    u = z - DV * t
    return u
end


@everywhere function ipmstep(x::Array{Float64},
    dx::Array{Float64},
    tau::Float64)

    neg = dx .< 0
    if sum(neg) == 0
        beta = tau;
    else
        x = x[neg]
        dx = dx[neg]
        dx = x ./ dx
        beta = tau * findmin(-dx)[1]
        beta = min(1.0, beta)
    end
    return beta
end


function ipmstep(x::SparseMatrixCSC{Float64,Int64},
    dx::Array{Float64,2},
    tau::Float64)

    neg = dx .< 0
    if sum(neg) == 0
        beta = tau;
    else
        x = x[neg]
        dx = dx[neg]
        dx = x ./ dx
        beta = tau * findmin(-dx)[1]
        beta = min(1.0, beta)
    end
    return beta
end


"""
    svm_ipm_dual(X::Array{Float64,2},
        y::Array{Float64,2},
        C::Float64,
        kernel::SVM_kernel,
        tol_ichol::Float64,
        maxdim::Int64,
        tol_ipm::Float64,
        maxiter::Int64,
        parallel::Bool)

Mehrotra IPM method for solving SVM with iChol Kernel approximations.

`X` : train data characteristics.
`Y` : train data labels.
`C` : penalization weight.
`kernel` : kernel type.
`tol_ichol` : ichol approximation tolerance.
`maxdim` : max dimension of the approximation matrix.
`tol_ipm` : dual gap tolerance.
`maxiter` : maximum number of iterations.
`parallel` : boolean for parallelization.
"""
function svm_ipm_dual(X::Array{Float64,2},
    y::Array{Float64,2},
    C::Float64,
    kernel::SVM_kernel,
    tol_ichol::Float64,
    maxdim::Int64,
    tol_ipm::Float64,
    maxiter::Int64,
    parallel::Bool)

    SUPPORT_VECTOR_TOL = 1e-5

    # Initial values
    # ---------------
    n, m = size(X)
    iter = 1

    alpha = ones(n, 1)
    if C == 1.0
        alpha /= 2
    end
    b = ones(1, 1)
    s = ones(n, 1)
    xi = ones(n, 1)
    C_alpha = C - alpha

    mu = (alpha' * s + C_alpha' * xi)[1]
    mu /= 2*n
    tol_ipm *= 1 + mu

    # Kernel Matrix approximation
    # ---------------------------
    if parallel
      V = distributed_kernel_ichol(X, y, kernel, tol_ichol, maxdim)
    else
      V = kernel_ichol(X, y, kernel, tol_ichol, maxdim)
    end

    @printf "  iter      mu           sigma           beta \n"
    @printf "---------------------------------------------------- \n"
    @printf " %3i     %1.4e     %1.4e     %1.4e  \n" iter mu 0.0 0.0



    while mu > tol_ipm && iter < maxiter

        # Solve predictor equations
        # --------------------------
        D = s ./ alpha + xi ./ C_alpha
        r = -1 - b[1] * y - s + xi - C * xi ./ C_alpha
        QDy = smw_solve(D, V, y)
        QDr = smw_solve(D, V, r)

        D_b = (y' * QDr) / (y' * QDy)
        D_alpha = -alpha + QDy * D_b - QDr
        D_xi = (xi ./ C_alpha) .* (D_alpha - C_alpha)
        D_s = (-s ./ alpha) .* (D_alpha + alpha)

        # Compute step
        # -------------
        B_alpha = ipmstep(alpha, D_alpha, 1.0)
        B_s = ipmstep(s, D_s, 1.0)
        B_xi = ipmstep(xi, D_xi, 1.0)

        # Compute sigma
        # --------------
        mu_aff = (s + B_s * D_s)' * (alpha + B_alpha * D_alpha)
        mu_aff += (C_alpha - B_alpha * D_alpha)' * (xi + B_xi * D_xi)
        mu_aff /= 2 * n
        mu_aff = mu_aff[1]
        sigma = (mu_aff / mu)^3

        # Solve corrector equations
        # -------------------------
        sig_mu = sigma * mu
        A_sig_mu = sig_mu ./ alpha
        A_da_ds = D_alpha .* D_s ./ alpha
        CA_sig_mu = sig_mu ./ C_alpha
        CA_da_dxi = D_alpha .* D_xi ./ C_alpha

        w1 = A_sig_mu - A_da_ds
        w2 = CA_sig_mu + CA_da_dxi
        w = w1 - w2
        u = r - w
        QDy = smw_solve(D, V, y)
        QDu = smw_solve(D, V, u)

        D_b = (y' * QDu) / (y' * QDy)
        D_alpha = -alpha + QDy * D_b - QDu
        D_xi = (xi ./ (C_alpha)) .* (D_alpha - C_alpha) + w2
        D_s = (-s ./ (alpha)) .* (D_alpha + alpha) + w1

        # Compute step
        # -------------
        B_alpha = ipmstep(alpha, D_alpha, 0.995)
        B_s = ipmstep(s, D_s, 0.995)
        B_xi = ipmstep(xi, D_xi, 0.995)
        B, ~ = findmin([B_alpha, B_s, B_xi])

        # Update
        # -------
        alpha += B_alpha * D_alpha
        b += B * D_b
        s += B_s * D_s
        xi += B_xi * D_xi
        C_alpha = C - alpha
        mu = (alpha' * s + C_alpha' * xi)[1]
        mu /= 2*n
        iter += 1

        @printf " %3i     %1.4e     %1.4e     %1.4e  \n" iter mu sigma B

    end

    # Create the predictor
    # ---------------------
    sv_index = alpha .> SUPPORT_VECTOR_TOL
    weights = alpha[sv_index]
    alpha = 0
    support_vectors = X[sv_index, :]
    X = 0
    support_vector_labels = y[sv_index, :]
    y = 0
    bias = 0.0

    predictor = SVM_predictor(
        kernel,
        C,
        bias,
        weights,
        support_vectors,
        support_vector_labels
    )
    predictor = compute_bias(predictor)

    return predictor

end


@everywhere function _compute_HDZ(H::Array, Dz::Array)
    n, m = size(H)
    col = Dz .* H
    col = [sum(col[:, i]) for i=1:m]
    return Array{Float64}(col)
end


@everywhere function _compute_HDH(H::Array, d::Array)
    DH = H ./ d
    return H' * DH
end


@everywhere function _compute_u(H::Array, d::Array, Dz::Array,
    t2::Array, u::Array)
    n = length(u)
    u[1:n] = Dz - H * t2 ./ d
end


"""
    parallel_smw(d::DArray, H::DArray, z::DArray)

This method solves the problem:
(D + HH^T) u = z
Where D is a diagonal matrix with values in `d`.
The problem is solved using Sherman-Morrison-Woodburry formula in parallel.
"""
function parallel_smw(d::DArray, H::DArray, z::DArray)
    n, m = size(H)

    pids = H.pids

    # Compute D^{-1}z
    Dz = z ./ d
    # Compute t1 = H'D^{-1}z
    refs = [@spawnat p _compute_HDZ(localpart(H), localpart(Dz)) for p in pids]
    t1 = pmap(fetch, refs)
    t1 = reduce(+, t1)
    # compute GG = I + H'D^{-1}H (parallel - G local)
    refs = [@spawnat p _compute_HDH(localpart(H), localpart(d)) for p in pids]
    GG = pmap(fetch, refs)
    GG = reduce(+, GG)
    GG += eye(m)
    # compute t2 (local - distribute t2)
    t2 = GG \ t1
    # Compute D^{-1}Ht2 ( distribute )
    u = dzeros(n)
    [@spawnat p _compute_u(
        localpart(H),
        localpart(d),
        localpart(Dz),
        t2,
        localpart(u)
    ) for p in pids]
    # return Dz - DHt2
    return u
end


"""
    parallel_step(x::DArray, dx::DArray, tau::Float64)

This method computes the step Beta, such that:
    x + Beta * dx > 0
"""
function parallel_step(x::DArray, dx::DArray, tau::Float64)
    pids = x.pids
    refs = [@spawnat p ipmstep(localpart(x), localpart(dx), tau) for p in pids]
    beta = pmap(fetch, refs)
    beta = reduce(min, beta)
    return beta
end


"""
    parallel_svm_ipm(X::Array{Float64,2},
        y::Array{Float64,2},
        C::Float64,
        kernel::SVM_kernel,
        tol_ichol::Float64,
        maxdim::Int64,
        tol_ipm::Float64,
        maxiter::Int64)

Mehrotra IPM method for solving SVM with iChol Kernel approximations.

`X` : train data characteristics.
`Y` : train data labels.
`C` : penalization weight.
`kernel` : kernel type.
`tol_ichol` : ichol approximation tolerance.
`maxdim` : max dimension of the approximation matrix.
`tol_ipm` : dual gap tolerance.
`maxiter` : maximum number of iterations.
"""
function parallel_svm_ipm(X::Array{Float64,2},
    y::Array{Float64,2},
    C::Float64,
    kernel::SVM_kernel,
    tol_ichol::Float64,
    maxdim::Int64,
    tol_ipm::Float64,
    maxiter::Int64)

    SUPPORT_VECTOR_TOL = 1e-5

    # Initial values
    # ---------------
    n, m = size(X)
    V = distributed_kernel_ichol(X, y, kernel, tol_ichol, maxdim)
    V = distribute(full(V))
    X = distribute(X)
    y = Array{Float64}([y[i] for i = 1:n])
    y = distribute(y)
    iter = 1

    pids = X.pids
    indexes = X.indexes
    N = length(pids)

    alpha = ones(n)
    if C <= 1.0
        alpha = dfill(C/2, n)
    else
        alpha = dones(n)
    end
    b = ones(1)
    s = dones(n)
    xi = dones(n)
    C_alpha = (C - alpha) * -1  # Bug in DistributedArrays

    mu = dot(alpha, s) + dot(C_alpha, xi)
    mu /= 2*n
    tol_ipm *= 1 + mu

    @printf "  iter      mu           sigma           beta \n"
    @printf "---------------------------------------------------- \n"
    @printf " %3i     %1.4e     %1.4e     %1.4e  \n" iter mu 0.0 0.0

    A_sig_mu = dzeros(n)
    CA_sig_mu = dzeros(n)

    while mu > tol_ipm && iter < maxiter

        # Solve predictor equations
        # --------------------------
        D = s ./ alpha + xi ./ C_alpha
        r = 1 -(- b[1] * y - s + xi - C * xi ./ C_alpha)
        QDy = parallel_smw(D, V, y)
        QDr = parallel_smw(D, V, r)

        D_b = Array{Float64}([dot(y, QDr) / dot(y, QDy)])
        D_alpha = D_b[1] * QDy - QDr - alpha
        D_xi = (xi ./ C_alpha) .* (D_alpha - C_alpha)
        D_s = -1 * (s ./ alpha) .* (D_alpha + alpha)

        # Compute step
        # -------------
        B_alpha = parallel_step(alpha, D_alpha, 1.0)
        B_s = parallel_step(s, D_s, 1.0)
        B_xi = parallel_step(xi, D_xi, 1.0)

        # Compute sigma
        # --------------
        mu_aff = dot(s + B_s * D_s, alpha + B_alpha * D_alpha)
        mu_aff += dot(C_alpha - B_alpha * D_alpha, xi + B_xi * D_xi)
        mu_aff /= 2 * n
        sigma = (mu_aff / mu)^3

        # Solve corrector equations
        # -------------------------
        sig_mu = sigma * mu
        A_sig_mu *= 0
        A_sig_mu += sig_mu
        A_sig_mu = A_sig_mu ./ alpha
        A_da_ds = D_alpha .* D_s ./ alpha
        CA_sig_mu *= 0
        CA_sig_mu += sig_mu
        CA_sig_mu = CA_sig_mu ./ C_alpha
        CA_da_dxi = D_alpha .* D_xi ./ C_alpha

        w1 = A_sig_mu - A_da_ds
        w2 = CA_sig_mu + CA_da_dxi
        w = w1 - w2
        u = r - w
        QDy = parallel_smw(D, V, y)
        QDu = parallel_smw(D, V, u)

        D_b = Array{Float64}([dot(y, QDu) / dot(y, QDy)])
        D_alpha = D_b[1] * QDy - QDu - alpha
        D_xi = (xi ./ (C_alpha)) .* (D_alpha - C_alpha) + w2
        D_s = -1 * (s ./ (alpha)) .* (D_alpha + alpha) + w1

        # Compute step
        # -------------
        B_alpha = parallel_step(alpha, D_alpha, 0.995)
        B_s = parallel_step(s, D_s, 0.995)
        B_xi = parallel_step(xi, D_xi, 0.995)
        B = min(B_alpha, B_s, B_xi)

        # Update
        # -------
        alpha += B_alpha * D_alpha
        b += B * D_b
        s += B_s * D_s
        xi += B_xi * D_xi
        C_alpha = -1 * (C - alpha)
        mu = dot(alpha, s) + dot(C_alpha, xi)
        mu /= 2*n
        iter += 1

        @printf " %3i     %1.4e     %1.4e     %1.4e  \n" iter mu sigma B

    end

    # Convert to local data
    alpha = convert(Array, alpha)
    X = convert(Array, X)
    y = convert(Array, y)

    # Create the predictor
    # ---------------------
    sv_index = alpha .> SUPPORT_VECTOR_TOL
    weights = alpha[sv_index]
    alpha = 0
    support_vectors = X[sv_index, :]
    X = 0
    support_vector_labels = y[sv_index, :]
    y = 0
    bias = 0.0

    predictor = SVM_predictor(
        kernel,
        C,
        bias,
        weights,
        support_vectors,
        support_vector_labels
    )
    predictor = compute_bias(predictor)

    return predictor

end
