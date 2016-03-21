#=
    This module...

    José Manuel Proudinat Silva
    jmps2812@gmail.com
    2016

=#

@everywhere using DistributedArrays
using Gadfly

include("types.jl")


@everywhere function K(x::Array{Float64},
                       y::Array{Float64},
                       kernel::SVM_kernel)
    #=
    Compute the kernel function between 'x' and 'y'.
    =#

    if kernel.kernel == "gaussian"
        return exp(-kernel.arg1 * norm(x - y))

    elseif kernel.kernel == "linear"
        return (x * y')[1]

    else
        return 0.0
    end

end


function compute_bias(predictor::SVM_predictor)
    predictor.bias = 0
    M = length(predictor.weights)
    bias = 0
    for i = 1:M
        bias += (
            predictor.support_vector_labels[i] -
            predict_value(
                predictor,
                predictor.support_vectors[i, :]
            )
        )
    end
    bias /= M
    predictor.bias = bias
    return predictor
end


function predict_value(predictor::SVM_predictor,
                       x::Array{Float64,2})
    result = predictor.bias
    M = length(predictor.weights)
    for i = 1:M
        result += (
            predictor.weights[i] *
            predictor.support_vector_labels[i] *
            K(predictor.support_vectors[i, :], x,
              predictor.kernel)
        )
    end
    return result
end


function predict(predictor::SVM_predictor,
                 x::Array{Float64,2})
    value = predict_value(predictor, x)
    return sign(value) * 1.0
end


function predict_matrix(predictor::SVM_predictor,
                        X::Array{Float64,2})
    n, ~ = size(X)
    preds = [predict(predictor, X[i, :]) for i=1:n]
    return preds
end


include("ichol.jl")
include("ipm.jl")
