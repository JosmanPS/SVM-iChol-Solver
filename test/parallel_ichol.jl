@everywhere using DistributedArrays

include("types.jl")
include("svm.jl")
include("ichol.jl")

X = distribute(rand(10000, 20));
XX = convert(Array, X);
sX = convert(SharedArray, XX);

Y = dones(10000, 1);
v = dzeros(10000, 1);
kernel = SVM_kernel("linear", 1.0, 1.0);

# diagonal kernel
@time parallel_kernel_diag(X, kernel, v);
@time vv = [K(XX[i, :], XX[i, :], kernel) for i in 1:n];
print(v)
