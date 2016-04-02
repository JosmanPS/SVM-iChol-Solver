
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
    n, m = size(V)

    pids = V.pids

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


function parallel_svm_ipm(X::Array{Float64,2},
                          y::Array{Float64,2},
                          C::Float64,
                          kernel::SVM_kernel,
                          tol_ichol::Float64,
                          maxdim::Int64,
                          tol_ipm::Float64,
                          maxiter::Int64)
    #=

    Description:
    ------------
    Mehrotra IPM method for solving SVM with
    iChol Kernel approximations.

    Input:
    ------
        - X : train data characteristics.
        - Y : train data labels.
        - C : penalization weight.
        - kernel : kernel type.
        - tol_ichol : ichol approximation tolerance.
        - maxdim : max dimension of the approximation matrix.
        - tol_ipm : dual gap tolerance.
        - maxiter : maximum number of iterations.

    Output:
    -------
        - predictor : SVM_predictor | trained SVM predictor.

    =#

    SUPPORT_VECTOR_TOL = 1e-5

    # Initial values
    # ---------------
    n, m = size(X)
    X = distribute(X)
    y = distribute(y)
    iter = 1

    pids = X.pids
    indexes = X.indexes
    N = length(pids)

    alpha = ones(n, 1)
    if C == 1.0
        alpha = dfill(0.5, n)
    else
        alpha = dones(n)
    end
    b = dones(1)
    s = dones(n)
    xi = dones(n)
    C_alpha = C - alpha
    mu = (alpha' * s + C_alpha' * xi)[1]
    mu = (alpha' * s + C_alpha' * xi)[1]

    mu = dot(alpha, s) + dot(C_alpha, xi)
    mu /= 2*n
    tol_ipm *= 1 + mu

    # Kernel Matrix approximation
    # ---------------------------
    V = distributed_kernel_ichol(X, y, kernel, tol_ichol, maxdim)
    V = distribute(V)


    @printf "  iter      mu           sigma           beta \n"
    @printf "---------------------------------------------------- \n"
    @printf " %3i     %1.4e     %1.4e     %1.4e  \n" iter mu 0.0 0.0



    while mu > tol_ipm && iter < maxiter

        # Solve predictor equations
        # --------------------------
        D = s ./ alpha + xi ./ C_alpha
        r = -1 - b[1] * y - s + xi - C * xi ./ C_alpha
        QDy = parallel_smw(D, V, y)
        QDr = parallel_smw(D, V, r)

        D_b = Array{Float64}([dot(y, QDr) / dot(y, QDy)])
        D_alpha = -alpha + D_b[1] * QDy - QDr
        D_xi = (xi ./ C_alpha) .* (D_alpha - C_alpha)
        D_s = (-s ./ alpha) .* (D_alpha + alpha)

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
        A_sig_mu = sig_mu ./ alpha
        A_da_ds = D_alpha .* D_s ./ alpha
        CA_sig_mu = sig_mu ./ C_alpha
        CA_da_dxi = D_alpha .* D_xi ./ C_alpha

        w1 = A_sig_mu - A_da_ds
        w2 = CA_sig_mu + CA_da_dxi
        w = w1 - w2
        u = r - w
        QDy = parallel_smw(D, V, y)
        QDu = parallel_smw(D, V, u)

        D_b = Array{Float64}([dot(y, QDu) / dot(y, QDy)])
        D_alpha = -alpha + D_b[1] * QDy - QDu
        D_xi = (xi ./ (C_alpha)) .* (D_alpha - C_alpha) + w2
        D_s = (-s ./ (alpha)) .* (D_alpha + alpha) + w1

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
        C_alpha = C - alpha
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
