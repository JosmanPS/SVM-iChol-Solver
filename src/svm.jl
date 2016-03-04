#=
    This module...

    José Manuel Proudinat Silva
    jmps2812@gmail.com
    2016

=#

module SVM

@everywhere using DistributedArrays

include("types.jl")
include("ichol.jl")
include("ipm.jl")

end