
#=
    SUPPORT VECTOR MACHINES TYPES
    =============================

    José Manuel Proudinat Silva
    jmps2812@gmail.com
    2016

=#


@everywhere type SVM_train_data
    X::Array{Float64}
    Y::Array{Float64}
end


@everywhere type SVM_test_data
    X::Array{Float64}
    Y::Array{Float64}
end


@everywhere type SVM_kernel
    kernel::AbstractString
    arg1::Float64
    arg2::Float64
end


@everywhere type SVM_predictor
    kernel::SVM_kernel
    C::Float64
    bias::Float64
    weights::Array{Float64}
    support_vectors::Array{Float64}
    support_vectors_labels::Array{Float64}
end


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
