#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FMIFlux

using FMI
include("FMI_neural.jl")
include("FMI_plot.jl")
#include("FMI_cache.jl")
include("misc.jl")

# FMI2_neural.jl
export fmi2DoStepME, fmi2DoStepCS
export fmi2InputDoStepCSOutput
export ME_NeuralFMU, CS_NeuralFMU, NeuralFMU, NeuralFMUInputLayer, NeuralFMUOutputLayer

# FMI_neural.jl
export fmiDoStepME, fmiDoStepCS
export fmiInputDoStepCSOutput
#export NeuralFMUCacheTime, NeuralFMUCacheState

# misc.jl
export mse_interpolate, transferParams!

end # module
