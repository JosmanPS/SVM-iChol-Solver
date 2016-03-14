
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


function step(x::Array{Float64,2},
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

    # Initial values
    # ---------------
    n, m = size(X)
    iter = 1

    alpha = spzeros(n, 1)
    b = ones(1, 1)
    s = ones(n, 1)
    xi = ones(n, 1)
    C_alpha = C - alpha

    mu = C / 2
    tol_ipm *= 1 + mu

    # Kernel Matrix approximation
    # ---------------------------
    data = SVM_train_data(X, y)
    X = 0
    V = kernel_ichol(data, kernel, tol_ichol, maxdim)
    data = 0

    @printf "  iter      mu      sigma      beta \n"
    @printf "------------------------------------- \n"
    @printf " %3i     %1.4e     %1.4e     %1.4e  \n" iter mu 0.0 0.0



    while mu > tol_ipm && iter < maxiter

        # Solve predictor equations
        # --------------------------
        D = s ./ alpha + xi ./ C_alpha
        r = -1 - b * y - s + xi - C * xi ./ C_alpha
        QDy = smw_solve(D, V, y)
        QDr = smw_solve(D, V, r)

        D_b = (y' * QDr) / (y' * QDy)
        D_alpha = -alpha + QDy * D_b - QDr
        D_xi = (xi ./ C_alpha) .* (D_alpha - C_alpha)
        D_s = (-s ./ alpha) .* (D_alpha + alpha)

        # Compute step
        # -------------
        B_alpha = step(alpha, D_alpha, 1.0)
        B_s = step(s, D_s, 1.0)
        B_xi = step(xi, D_xi, 1.0)

        # Compute sigma
        # --------------
        mu_aff = (s + B_s * D_s)' * (alpha + B_alpha * D_alpha)
        mu_aff += (C_alpha - B_alpha * D_alpha)' * (xi + B_xi * D_xi)
        mu_aff /= 2 * n
        mu_aff = mu_aff[1]
        sigma = (mu_aff / mu)^3

        # Solve corrector equations
        # -------------------------
        D = s ./ (alpha + D_alpha) + xi ./ (C_alpha - D_alpha)
        sig_mu = sigma * mu
        alpha_D_alpha = alpha + D_alpha
        C_alpha_D_alpha = C_alpha - D_alpha

        A_sig_mu = sig_mu ./ (alpha_D_alpha)
        CA_sig_mu = sig_mu ./ (C_alpha_D_alpha)
        w = A_sig_mu - CA_sig_mu
        r = -1 - b * y - s + xi - C * xi ./ (C_alpha_D_alpha)
        u = r - w
        QDy = smw_solve(D, V, y)
        QDu = smw_solve(D, V, u)

        D_b = (y' * QDu) / (y' * QDy)
        D_alpha_2 = -alpha + QDy * D_b - QDu
        D_xi = (xi ./ (C_alpha_D_alpha)) .* (D_alpha_2 - C_alpha) + CA_sig_mu
        D_s = (-s ./ (alpha_D_alpha)) .* (D_alpha_2 + alpha) + A_sig_mu
        D_alpha = D_alpha_2
        D_alpha_2 = 0

        # Compute step
        # -------------
        B_alpha = step(alpha, D_alpha, 0.995)
        B_s = step(s, D_s, 0.995)
        B_xi = step(xi, D_xi, 0.995)
        B, ~ = findmin([B_alpha, B_s, B_xi])

        # Update
        # -------
        alpha += B_alpha * D_alpha
        b += B * D_b
        s += B_s * D_s
        xi += B_xi * D_xi
        C_alpha = C - alpha
        mu = (alpha' * s + C_alpha * xi)[1]
        mu /= 2*n
        iter += 1

        @printf " %3i     %1.4e     %1.4e     %1.4e  \n" iter mu sigma B

    end    

    return (alpha, b, s, xi)

end