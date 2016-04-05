
using Gadfly

include("../src/psvm.jl")
srand(130056)

P = length(workers());
kernel = SVM_kernel("linear", 1.0, 1.0);

s_times = zeros(5);
p_times = zeros(5);
labels = vcat(
    ["serial" for i = 1:5],
    ["parallel" for i = 1:5]
);

X = rand(10, 10);
Y = ones(10, 1);
n = 10;
kernel_ichol(X, Y, kernel, 1e-8, n);
distributed_kernel_ichol(X, Y, kernel, 1e-8, n);

for i = 1:5
    n = 10 ^ i;
    X = rand(1000, n);
    Y = ones(1000, 1);

    # tic();
    # kernel_ichol(X, Y, kernel, 1e-8, n);
    # s_times[i] = toq();

    tic();
    distributed_kernel_ichol(X, Y, kernel, 1e-8, n);
    p_times[i] = toq();
end

times = vcat(s_times, p_times);
n = vcat(
    collect(1:5),
    collect(1:5)
);

plot(
    layer(x=n, y=times, color=labels, Geom.line, order=1),
    layer(x=n, y=times, color=labels, Geom.point, order=2),
);

plot(
    layer(x=n, y=log(times), color=labels, Geom.line, order=1),
    layer(x=n, y=log(times), color=labels, Geom.point, order=2),
);
