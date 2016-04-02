
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

df = readtable("../datasets/lfg.csv");
x1 = convert(Array{Float64}, df.columns[2]);
x2 = convert(Array{Float64}, df.columns[3]);
X = hcat(x1, x2);
n, m = size(X);
Y = convert(Array{Float64}, df.columns[4]) .* ones(n, 1);

rsorting = sample(1:n, n, replace=false);
X = X[rsorting, :]
Y = Y[rsorting, :]

kernel = SVM_kernel("gaussian", 1.0, 1.0);

function string_array(x::Array{Float64})
  n = length(x)
  x = [string(x[i]) for i=1:n]
  return x
end

grid = [0.0 0.0];
for i=-5:0.1:7
  for j=-4:0.1:4
    grid = vcat(grid, [i j]);
  end
end


predictor = svm_ipm_dual(
    X,
    Y,
    10.0,
    kernel,
    1e-8,
    600,
    1e-8,
    50,
    true
);


preds = predict_matrix(predictor, X);
preds = preds .== Y;
preds = sum(preds) / n;
print("Accuracy: ", preds, "\n\n")

grid_labels = predict_matrix(predictor, grid);

plot(layer(x=grid[:,1], y=grid[:,2], color=string_array(grid_labels[:,1]), Geom.point, Theme(default_point_size=Measures.Measure(0.4mm)),order=1),
     layer(x=X[:,1], y=X[:,2], color=string_array(Y[:,1]), Geom.point, order=2),
)
