#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FMIFlux

using FMI
include("FMI_neural.jl")
include("FMI_plot.jl")

# FMI2_neural.jl
export fmi2DoStepME
export fmi2InputDoStepCSOutput, fmi2InputDoStepMEOutput, fmi2InputDoStepME
export ME_NeuralFMU, CS_NeuralFMU, NeuralFMU, NeuralFMUInputLayer, NeuralFMUOutputLayer

export NeuralFMUCacheTime, NeuralFMUCacheState

end # module
