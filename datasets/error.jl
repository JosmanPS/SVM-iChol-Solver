
include("../src/psvm.jl")

function readlibsvm(path::AbstractString)
    data = readdlm(path, ',')[2:end, 2:end]
    data = Array{Float64}(data)
    Y = data[:, 1]
    Y = Y .* ones(length(Y), 1)
    X = data[:, 2:end]
    return X, Y
end

function svmdata(path::AbstractString)
    if path == "./lfg.csv"
        data = readdlm(path, ',')[2:end, 2:end]
        data = Array{Float64}(data)
        Y = data[:, 3]
        Y = Y .* ones(length(Y), 1)
        X = data[:, 1:2]
        return X, Y
    elseif path == "./wwqc.csv"
        data = readdlm(path, ',')
        data = Array{Float64}(data)
        Y = data[:, 1]
        Y = Y .* ones(length(Y), 1)
        X = data[:, 2:end]
        return X, Y
    else
        return readlibsvm(path)
    end
end

datasets = [
    "./lfg.csv",
    "./a1a.csv",
    "./a5a.csv",
    "./a6a.csv",
    "./diabetes.csv",
    "./german.csv",
    "./madelon.csv",
    "./wwqc.csv"
]

N = length(datasets)

s_times = zeros(N)
p_times = zeros(N)

kernel = SVM_kernel("gaussian", 1.0, 1.0)
C = 10.0
tols = [0.2, 0.4, 0.6, 0.8]
tol_ipm = 1e-4
maxiter = 100
parallel = false
error = zeros(4)
local_error = zeros(N)


for j = 1:length(tols)
    tol = tols[j]
    for i = 1:N
        path = datasets[i]
        X, Y = svmdata(path)
        n, m = size(X)
        rsorting = sample(1:n, n, replace=false);
        X = X[rsorting, :]
        Y = Y[rsorting, :]
        maxdim = min(n, m)
        # tic()
        # svm_ipm_dual(X, Y, C, kernel, tol_ichol,
        #     maxdim, tol_ipm, maxiter, parallel)
        # s_times[i] = toq()
        predictor = parallel_svm_ipm(X, Y, C, kernel, tol,
            maxdim, tol_ipm, maxiter)
        preds = predict_matrix(predictor, X);
        preds = preds .== Y;
        preds = sum(preds) / n;
        local_error[i] = preds
    end
    error[j] = mean(local_error)
end
