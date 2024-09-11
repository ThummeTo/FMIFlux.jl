#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FMIFlux
import FMISensitivity

import FMISensitivity.ForwardDiff
import FMISensitivity.Zygote
import FMISensitivity.ReverseDiff
import FMISensitivity.FiniteDiff

@debug "Debugging messages enabled for FMIFlux ..."

if VERSION < v"1.7.0"
    @warn "Training under Julia 1.6 is very slow, please consider using Julia 1.7 or newer." maxlog =
        1
end

import FMIImport.FMIBase: hasCurrentInstance, getCurrentInstance, unsense
import FMISensitivity.ChainRulesCore: ignore_derivatives

import FMIImport

import Flux

using FMIImport

include("optimiser.jl")
include("hotfixes.jl")
include("convert.jl")
include("flux_overload.jl")
include("neural.jl")
include("misc.jl")
include("layers.jl")
include("deprecated.jl")
include("batch.jl")
include("losses.jl")
include("scheduler.jl")
include("compatibility_check.jl")

# optional extensions
using FMIImport.FMIBase.Requires
using FMIImport.FMIBase.PackageExtensionCompat
function __init__()
    @require_extensions
end

# JLD2.jl
function saveParameters end
function loadParameters end

# FMI_neural.jl
export ME_NeuralFMU, CS_NeuralFMU, NeuralFMU

# misc.jl
export mse_interpolate, transferParams!, transferFlatParams!, lin_interp

# scheduler.jl
export WorstElementScheduler,
    WorstGrowScheduler, RandomScheduler, SequentialScheduler, LossAccumulationScheduler

# batch.jl 
export batchDataSolution, batchDataEvaluation

# layers.jl
# >>> layers are exported inside the file itself

# deprecated.jl
# >>> deprecated functions are exported inside the file itself

end # module
