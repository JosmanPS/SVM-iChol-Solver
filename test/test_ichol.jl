dim = 50
instances = 100000
X = rand(instances, dim);
# X = convert(SharedArray, X);
Y = ones(instances, 1);
# Y = convert(SharedArray, Y);

# data = SVM_train_data(X, Y);
@everywhere kernel = SVM_kernel("linear", 1, 1);

#=
Q = X*X';
@time T = kernel_ichol(data, kernel, 0.05);
@time T = parallel_kernel_ichol(data, kernel, 0.05);
err = norm(Q - T*T', Inf)
err_rel = err / norm(Q, Inf)
=#