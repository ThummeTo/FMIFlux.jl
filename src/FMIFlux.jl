#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FMIFlux

@debug "Debugging messages enabled for FMIFlux ..."

if VERSION < v"1.7.0"
    @warn "Training under Julia 1.6 is very slow, please consider using Julia 1.7 or newer."
end

# Overwrite tag printing and limit partials length from ForwardDiff.jl 
# import FMIImport.ForwardDiff
# function Base.show(io::IO, d::ForwardDiff.Dual{T,V,N}) where {T,V,N}
#     print(io, "Dual(", ForwardDiff.value(d))
#     for i in 1:min(N, 5)
#         print(io, ", ", ForwardDiff.partials(d, i))
#     end
#     if N > 5
#         print(io, ", [$(N-5) more]...")
#     end
#     print(io, ")")
# end

# ToDo: Quick-fixes until patch release SciMLSensitivity v0.7.29
import FMIImport.SciMLSensitivity: FakeIntegrator, u_modified!, TrackedAffect
import FMIImport.SciMLSensitivity.DiffEqBase: set_u!
function u_modified!(::FakeIntegrator, ::Bool)
    return nothing
end
function set_u!(::FakeIntegrator, u)
    return nothing
end

# ToDo: Quick-fixes until patch release SciMLSensitivity v0.7.28
# function Base.hasproperty(f::TrackedAffect, s::Symbol)
#     if hasfield(TrackedAffect, s)               
#         return true
#     else
#         _affect = getfield(f, :affect!)
#         return hasfield(typeof(_affect), s)
#     end
# end
# function Base.getproperty(f::TrackedAffect, s::Symbol)
#     if hasfield(TrackedAffect, s)               
#         return getfield(f, s)
#     else
#         _affect = getfield(f, :affect!)
#         return getfield(_affect, s)
#     end
# end
# function Base.setproperty!(f::TrackedAffect, s::Symbol, value)
#     if hasfield(TrackedAffect, s)               
#         return setfield!(f, s, value)
#     else
#         _affect = getfield(f, :affect!)
#         return setfield!(_affect, s, value)
#     end
# end

using Requires, Flux 

using FMIImport
using FMIImport: fmi2ValueReference, FMU, FMU2, FMU2Component
using FMIImport: fmi2Struct
using FMIImport: fmi2SetupExperiment, fmi2EnterInitializationMode, fmi2ExitInitializationMode, fmi2Reset, fmi2Terminate
using FMIImport: fmi2NewDiscreteStates, fmi2SetContinuousStates, fmi2GetContinuousStates, fmi2GetNominalsOfContinuousStates
using FMIImport: fmi2SetTime, fmi2CompletedIntegratorStep, fmi2GetEventIndicators, fmi2GetDerivatives, fmi2GetReal
using FMIImport: fmi2SampleJacobian, fmi2GetDirectionalDerivative, fmi2GetJacobian, fmi2GetJacobian!
using FMIImport: fmi2True, fmi2False

import FMIImport.FMICore: fmi2ValueReferenceFormat

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
