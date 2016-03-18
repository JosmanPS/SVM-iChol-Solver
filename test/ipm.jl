using Gadfly

# Random data
# --------------
num_samples = 100;
num_features = 2;

samples = randn((num_samples, num_features));
labels = 2 * (sum(samples, 2) .> 0) - 1.0;

#
kernel = SVM_kernel("linear", 1.0, 1.0);
alpha = svm_ipm_dual(
    samples,
    labels,
    1000.0,
    kernel,
    1e-8,
    2,
    1e-8,
    50
);

w = samples' * alpha[1];
b = alpha[2];
base = [i for i=-2:0.1:2];
basey = b[1] + w[1] * base;
basey /= w[2];

plot(layer(x=samples[:,1], y=samples[:,2], color=labels, Geom.point),
     layer(x=base, y=basey, Geom.line)
)