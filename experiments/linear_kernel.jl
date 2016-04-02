@everywhere using DistributedArrays
using DataFrames
using Gadfly

include("../src/types.jl")
include("../src/svm.jl")
include("../src/ichol.jl")
include("../src/parallel_ichol.jl")
include("../src/ipm.jl")
# include("./src/parallel_ipm.jl")

srand(130056)

num_samples = 100;
num_features = 2;

samples = randn((num_samples, num_features));
labels = 2 * (sum(samples, 2) .> 0) - 1.0;

function string_array(x::Array{Float64})
  n = length(x)
  x = [string(x[i]) for i=1:n]
  return x
end

kernel = SVM_kernel("linear", 1.0, 1.0);

grid = [0.0 0.0];
for i=-3:0.1:3
  for j=-3:0.1:3
    grid = vcat(grid, [i j]);
  end
end

predictor = svm_ipm_dual(
    samples,
    labels,
    100.0,
    kernel,
    1e-8,
    1,
    1e-8,
    50,
    false
);

preds = predict_matrix(predictor, samples);
preds = preds .== labels;
preds = sum(preds) / num_samples;
print("Accuracy: ", preds, "\n\n")

grid_labels = predict_matrix(predictor, grid);

plot(layer(x=grid[:,1], y=grid[:,2], color=string_array(grid_labels[:,1]), Geom.point, Theme(default_point_size=Measures.Measure(0.4mm)),order=1),
     layer(x=samples[:,1], y=samples[:,2], color=string_array(labels[:,1]), Geom.point, order=2),
)
