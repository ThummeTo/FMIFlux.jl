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
    @warn "Training under Julia 1.6 is very slow, please consider using Julia 1.7 or newer." maxlog=1
end

import FMIImport.FMICore: hasCurrentComponent, getCurrentComponent, unsense
import FMIImport.FMICore.ChainRulesCore: ignore_derivatives

import DifferentialEquations
import FMIImport

using Requires
import Flux

using FMIImport
using FMIImport: fmi2ValueReference, FMU, FMU2, FMU2Component
using FMIImport: fmi2Struct
using FMIImport: fmi2SetupExperiment, fmi2EnterInitializationMode, fmi2ExitInitializationMode, fmi2Reset, fmi2Terminate
using FMIImport: fmi2NewDiscreteStates, fmi2SetContinuousStates, fmi2GetContinuousStates, fmi2GetNominalsOfContinuousStates
using FMIImport: fmi2SetTime, fmi2CompletedIntegratorStep, fmi2GetEventIndicators, fmi2GetDerivatives, fmi2GetReal
using FMIImport: fmi2SampleJacobian, fmi2GetDirectionalDerivative, fmi2GetJacobian, fmi2GetJacobian!
using FMIImport: fmi2True, fmi2False

import FMIImport.FMICore: fmi2ValueReferenceFormat, fmi2Real

include("optimiser.jl")
include("hotfixes.jl")
include("convert.jl")
include("flux_overload.jl")
include("neural.jl")
#include("chain_rules.jl")
include("misc.jl")
include("layers.jl")
include("deprecated.jl")
include("batch.jl")
include("losses.jl")
include("scheduler.jl")
include("compatibility_check.jl")

# from Plots.jl 
# No export here, Plots.plot is extended if available.

# from FMI.jl 
function fmiPlot(nfmu::NeuralFMU; kwargs...)
    @assert false "fmiPlot(...) needs `Plots` package. Please install `Plots` and do `using Plots` or `import Plots`."
end
function fmiPlot!(fig, nfmu::NeuralFMU; kwargs...)
    @assert false "fmiPlot!(...) needs `Plots` package. Please install `Plots` and do `using Plots` or `import Plots`."
end
# No export here, FMI.fmiPlot is extended.

# from JLD2.jl
function fmiSaveParameters(nfmu::NeuralFMU, path::String; keyword="parameters")
    @assert false "fmiSaveParameters(...) needs `JLD2` package. Please install `JLD2` and do `using JLD2` or `import JLD2`."
end
function fmiLoadParameters(nfmu::NeuralFMU, path::String; flux_model=nothing, keyword="parameters")
    @assert false "fmiLoadParameters(...) needs `JLD2` package. Please install `JLD2` and do `using JLD2` or `import JLD2`."
end
export fmiSaveParameters, fmiLoadParameters

function __init__()
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin 
        import .Plots
        include("extensions/Plots.jl")
    end

    @require FMI="14a09403-18e3-468f-ad8a-74f8dda2d9ac" begin 
        @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin 
            import .FMI 
            include("extensions/FMI.jl")
        end
    end

    @require JLD2="033835bb-8acc-5ee8-8aae-3f567f8a3819" begin
        import .JLD2
        include("extensions/JLD2.jl")
    end
end

# FMI_neural.jl
export ME_NeuralFMU, CS_NeuralFMU, NeuralFMU

# misc.jl
export mse_interpolate, transferParams!, transferFlatParams!, lin_interp

# scheduler.jl
export WorstElementScheduler, WorstGrowScheduler, RandomScheduler, SequentialScheduler, LossAccumulationScheduler
export initialize!, update!

# batch.jl 
export batchDataSolution, batchDataEvaluation

# layers.jl
# >>> layers are exported inside the file itself

# deprecated.jl
# >>> deprecated functions are exported inside the file itself

end # module
