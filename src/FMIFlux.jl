#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FMIFlux

using FMI
include("FMI_neural.jl")
#include("FMI2_neural.jl")
include("FMI_plot.jl")

# FMI2_neural.jl
export fmi2DoStepME, NeuralFMU, NeuralFMUInputLayer, NeuralFMUOutputLayer

# FMI_plot.jl
# export fmiPlot

end # module
