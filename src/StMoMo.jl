module StMoMo

using LinearAlgebra
using Optim
using ForwardDiff
using ADTypes
using Statistics
using Distributions
using SpecialFunctions

genWeightMat = include("genWeightMat.jl")

export genWeightMat
end
