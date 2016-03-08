
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
    b = 1
    s = ones(n, 1)
    xi = ones(n, 1)

    mu = C / 2

    # Kernel Matrix approximation
    # ---------------------------
    data = SVM_train_data(X, y)
    X = 0
    V = kernel_ichol(data, kernel, tol_ichol, maxdim)
    data = 0

    while mu > tol_ipm && iter < maxiter

      # Solve predictor equations
      # --------------------------
      C_alpha = C - alpha
      D = s ./ alpha + xi ./ C_alpha
      r = -1 - b * y - s + xi - C * xi ./ C_alpha
      QDy = smw_solve(D, V, y)
      QDr = smw_solve(D, V, r)

      D_b = QDr / (y' * QDy)
      D_alpha = -alpha + QDy * D_b - QDr
      D_xi = (xi ./ C_alpha) .* (D_alpha - C_alpha)
      D_s = (-s ./ alpha) .* (D_alpha - alpha)

      # Compute step

      # Compute sigma

      # Solve corrector equations

      # Compute step

      # Update

    end    


    return 0

end