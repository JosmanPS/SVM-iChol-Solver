
#=
    INTERIOR POINT METHODS FOR SVM
    ===============================

    José Manuel Proudinat Silva
    jmps2812@gmail.com
    2016

=#


function spdiag(d::Array{Float64})
    #=
      Creates a sparse diagonal matrix
      with values in d.
    =#
    n = length(d)
    D = spzeros(n, n)
    for i in 1:n
      D[i, i] = d[i]
    end
    return D
end


function smw_solve(d::Array{Float64,2},
                   V::Array{Float64,2},
                   w::Array{Float64,2})
    #=

    This method solves the problem:
      (D + VV')u = w
    where D is a diagonal matrix (diag(d));
    using the Sherman-Morrison-Woodburry formula.

    =#
    n, m = size(V)
    z = w ./ d
    DV = spdiag(1./d) * V
    A = speye(m) + V' * DV
    b = V' * z
    t = cholfact(A) \ b
    u = z - DV * t
    return u

end


function smw_solve(d::Array{Float64,2},
                   V::SparseMatrixCSC{Float64,Int64},
                   w::Array{Float64,2})
    #=

    This method solves the problem:
      (D + VV')u = w
    where D is a diagonal matrix (diag(d));
    using the Sherman-Morrison-Woodburry formula.

    =#
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


function ipmstep(x::Array{Float64,2},
                 dx::Array{Float64,2},
                 tau::Float64)
    #=

    This method compute the step Beta,
    such that:
        x + Beta * dx > 0

    =#
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
    #=

    This method compute the step Beta,
    such that:
        x + Beta * dx > 0

    =#
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


function svm_ipm_dual(X::Array{Float64,2},
                      y::Array{Float64,2},
                      C::Float64,
                      kernel::SVM_kernel,
                      tol_ichol::Float64,
                      maxdim::Int64,
                      tol_ipm::Float64,
                      maxiter::Int64)
    #=

    Mehrotra IPM method for solving SVM with
    iChol Kernel approximations.

    =#

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
    data = SVM_train_data(X, y)
    V = kernel_ichol(data, kernel, tol_ichol, maxdim)
    data = 0

    @printf "  iter      mu      sigma      beta \n"
    @printf "------------------------------------- \n"
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
        Valpha = V' * alpha
        OBJ = (Valpha' * Valpha)[1] - sum(alpha)
        @printf " OBJ: %1.8e \n" OBJ

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