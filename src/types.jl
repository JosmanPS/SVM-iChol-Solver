
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
    weights::Array{Float64,1}
    support_vectors::Array{Float64,2}
    support_vector_labels::Array{Float64,2}
end

