
# Random data
# --------------
num_samples = 100;
num_features = 2;

samples = randn((num_samples, num_features));
labels = 2 * (sum(samples, 2) .> 0) - 1.0;

#
kernel = SVM_kernel("linear", 1.0, 1.0);
predictor = svm_ipm_dual(
    samples,
    labels,
    100.0,
    kernel,
    1e-8,
    2,
    1e-8,
    50,
    false
);

grid = [0.0 0.0];
for i=-2:0.1:2
    for j=-2:0.1:2
        grid = vcat(grid, [i j]);
    end
end
grid_labels = predict_matrix(predictor, grid);


function string_array(x::Array{Float64})
    n = length(x)
    x = [string(x[i]) for i=1:n]
    return x
end

plot(layer(x=grid[:,1], y=grid[:,2], color=string_array(grid_labels[:,1]), Geom.point, Theme(default_point_size=Measures.Measure(0.4mm)),order=1),
     layer(x=samples[:,1], y=samples[:,2], color=string_array(labels[:,1]), Geom.point, order=2),
)

# distributed_kernel_ichol(samples, labels, kernel, 0.3, 14)
