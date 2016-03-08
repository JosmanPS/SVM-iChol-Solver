
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
    T = kernel_ichol(data, kernel, tol_ichol, maxdim)
    data = 0

    while mu > tol_ipm && iter < maxiter

      # Solve predictor equations

      # Compute step

      # Compute sigma

      # Solve corrector equations

      # Compute step

      # Update

    end    


    return 0

end