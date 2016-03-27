#=
    This module...

    José Manuel Proudinat Silva
    jmps2812@gmail.com
    2016

=#

# @everywhere using DistributedArrays
# using Gadfly

# include("types.jl")


@everywhere function K(x::Array{Float64},
                       y::Array{Float64},
                       kernel::SVM_kernel)
    #=

    Description:
    ------------
    Compute the kernel function between 'x' and 'y'.

    Input:
    ------
        - x : train data instance.
        - y : train data instance.
        - kernel : kernel type.

    Output:
    -------
        - Float64 | kernel function value.

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
    #=

    Description:
    ------------
    Compute the bias constant of the trained SVM predictor.

    Input:
    ------
        - predictor : trained SVM predictor.

    Output:
    -------
        - predictor : SVM_predictor | trained SVM predictor
                      with bias constant.

    =#

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
    #=

    Description:
    ------------
    Compute the prediction value of a data instance.
    i.e.    f(x) = sum( alpha_i * y_i * K(X_i, x_i) )

    Input:
    ------
        - predictor : trained SVM predictor.
        - x : data instance to predict.

    Output:
    -------
        - result : Float64 | prediction value.

    =#

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
    #=

    Description:
    ------------
    Compute the prediction value sign of a data instance.
    i.e.    f(x) = sign(sum( alpha_i * y_i * K(X_i, x_i) ))

    Input:
    ------
        - predictor : trained SVM predictor.
        - x : data instance to predict.

    Output:
    -------
        - Float64 | prediction value sign.

    =#

    value = predict_value(predictor, x)
    return sign(value) * 1.0

end


function predict_matrix(predictor::SVM_predictor,
                        X::Array{Float64,2})
    #=

    Description:
    ------------
    Compute the prediction value signs of a data matrix.

    Input:
    ------
        - predictor : trained SVM predictor.
        - X : data matrix to predict.

    Output:
    -------
        - Array{Float64,1} | prediction value signs.

    =#
    n, ~ = size(X)
    preds = [predict(predictor, X[i, :]) for i=1:n]
    return preds

end


include("ichol.jl")
include("ipm.jl")
