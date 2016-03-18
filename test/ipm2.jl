using Gadfly

num_samples = 10;
num_features = 2;

samples = randn((num_samples, num_features));
labels = 2 * (sum(samples, 2) .> 0) - 1.0;
kernel = SVM_kernel("linear", 1.0, 1.0);

X = samples
y = labels
C = 10
tol_ichol = 1e-8
maxdim = 2
tol_ipm = 1e-8
maxiter = 20


# --------------------------------------------------------------

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

D = s ./ alpha + xi ./ C_alpha
r = -1 - b[1] * y - s + xi - C * xi ./ C_alpha
QDy = smw_solve(D, V, y)
QDr = smw_solve(D, V, r)

D_b = (y' * QDr) / (y' * QDy)
D_alpha = -alpha + QDy * D_b - QDr
D_xi = (xi ./ C_alpha) .* (D_alpha - C_alpha)
D_s = (-s ./ alpha) .* (D_alpha + alpha)


# --------------------------------------------------------------

function diagonal(x)
    n = length(x)
    X = zeros(n, n)
    for i = 1:n
        X[i,i] = x[i]
    end
    return X
end

Q = V * V'
I = eye(n)
S = diagonal(s)
A = diagonal(alpha)
XI = diagonal(xi)
CA = diagonal(C_alpha)

ipm_M = hcat(
    vcat(Q, y', S, -XI),
    vcat(-y, [0], zeros(n,1), zeros(n,1)),
    vcat(-I, zeros(1,n), A, zeros(n,n)),
    vcat(I, zeros(1,n), zeros(n,n), CA)
)

F1 = Q*alpha - b[1]*y - s + xi - 1
F2 = alpha' * y
F3 = S*alpha
F4 = XI * C_alpha

F = vcat(F1, F2, F3, F4)

DELTA = ipm_M \ -F
grad_OBJ = Q * alpha - 1

# --------------------------------------------------------------

#
# Las ecuaciones del paso predictor funcionan correctamente
#


# Compute step
# -------------
B_alpha = ipmstep(alpha, D_alpha, 1.0)
B_s = ipmstep(s, D_s, 1.0)
B_xi = ipmstep(xi, D_xi, 1.0)

mu_aff = (s + B_s * D_s)' * (alpha + B_alpha * D_alpha)
mu_aff += (C_alpha - B_alpha * D_alpha)' * (xi + B_xi * D_xi)
mu_aff /= 2 * n
mu_aff = mu_aff[1]
sigma = (mu_aff / mu)^3



# --------------------------------------------------------------

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


# --------------------------------------------------------------



F3 += -sig_mu + DELTA[1:10] .* DELTA[12:21]
F4 += -sig_mu - DELTA[1:10] .* DELTA[22:31]

F = vcat(F1, F2, F3, F4)

DELTA = ipm_M \ -F

#
# La tercera y cuarta ecuación no coincide en el paso corrector
#
