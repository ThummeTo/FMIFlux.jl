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
export fmi2GetJacobian, fmi2GetJacobian!, fmi2GetFullJacobian, fmi2GetFullJacobian!

# FMI_neural.jl
export fmiDoStepME, fmiDoStepCS
export fmiInputDoStepCSOutput

# misc.jl
export mse_interpolate, transferParams!, lin_interp

# debugging only 
# export _build_jac_dx_x_slow

end # module
