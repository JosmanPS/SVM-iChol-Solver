
@everywhere using DistributedArrays
using DataFrames

include("../src/types.jl")
include("../src/svm.jl")
include("../src/ichol.jl")
include("../src/parallel_ichol.jl")
include("../src/ipm.jl")
include("../src/parallel_ipm.jl")

srand(130056)

kernel = SVM_kernel("gaussian", 1.0, 1.0);

df = readtable("../datasets/lfg.csv");
x1 = convert(Array{Float64}, df.columns[2]);
x2 = convert(Array{Float64}, df.columns[3]);
X = hcat(x1, x2);
n, m = size(X);
Y = convert(Array{Float64}, df.columns[4]) .* ones(n, 1);

rsorting = sample(1:n, n, replace=false);
X = X[rsorting, :]
Y = Y[rsorting, :]

H = kernel_ichol(X, Y, kernel, 1e-4, 20);
H = distributed_kernel_ichol(X, Y, kernel, 1e-4, 20);
