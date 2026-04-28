module StMoMo

using LinearAlgebra
using Optim
using ForwardDiff
using ADTypes
using Statistics
using Distributions
using SpecialFunctions

include("genWeightMat.jl")
include("lc.jl")
include("fit_StMoMo.jl")
export genWeightMat, fit_StMoMo
end
