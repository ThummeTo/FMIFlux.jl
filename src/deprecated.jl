#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

"""
DEPRECATED:

Performs something similar to `fmiDoStep` for ME-FMUs (note, that fmiDoStep is for CS-FMUs only).
Event handling (state- and time-events) is supported. If you don't want events to be handled, you can disable event-handling for the NeuralFMU `nfmu` with the attribute `eventHandling = false`.

Optional, additional FMU-values can be set via keyword arguments `setValueReferences` and `setValues`.
Optional, additional FMU-values can be retrieved by keyword argument `getValueReferences`.

Function takes the current system state array ("x") and returns an array with state derivatives ("x dot") and optionally the FMU-values for `getValueReferences`.
Setting the FMU time via argument `t` is optional, if not set, the current time of the ODE solver around the NeuralFMU is used.
"""
function fmi2EvaluateME(fmu::FMU2,
        x::Array{<:Real},
        t,#::Real,
        setValueReferences::Union{Array{fmi2ValueReference}, Nothing}=nothing,
        setValues::Union{Array{<:Real}, Nothing}=nothing,
        getValueReferences::Union{Array{fmi2ValueReference}, Nothing}=nothing) 

    y = nothing
    y_refs = getValueReferences
    u = setValues
    u_refs = setValueReferences
    
    if y_refs != nothing
        y = zeros(length(y_refs))
    end
    
    dx = zeros(length(x))

    c = fmu.components[end]

    y, dx = c(dx=dx, y=y, y_refs=y_refs, x=x, u=u, u_refs=u_refs, t=t)
    
    return [(dx == nothing ? [] : dx)..., (y == nothing ? [] : y)...]
end
export fmi2EvaluateME

"""
DEPRECATED:

Wrapper. Call ```fmi2EvaluateME``` for more information.
"""
function fmiEvaluateME(str::fmi2Struct, 
                     x::Array{<:Real}, 
                     t::Real = (typeof(str) == FMU2 ? str.components[end].t : str.t),
                     setValueReferences::Union{Array{fmi2ValueReference}, Nothing} = nothing, 
                     setValues::Union{Array{<:Real}, Nothing} = nothing, 
                     getValueReferences::Union{Array{fmi2ValueReference}, Nothing} = nothing )
    fmi2EvaluateME(str, x, t,
                setValueReferences,
                setValues,
                getValueReferences)
end
export fmiEvaluateME

"""
DEPRECATED:

Wrapper. Call ```fmi2DoStepCS``` for more information.
"""
function fmiDoStepCS(str::fmi2Struct, 
                     dt::Real,
                     setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0), 
                     setValues::Array{<:Real} = zeros(Real, 0),
                     getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    fmi2DoStepCS(str, dt, setValueReferences, setValues, getValueReferences)
end
export fmiDoStepCS

"""
DEPRECATED:

Wrapper. Call ```fmi2InputDoStepCSOutput``` for more information.
"""
function fmiInputDoStepCSOutput(str::fmi2Struct, 
                                dt::Real, 
                                u::Array{<:Real})
    fmi2InputDoStepCSOutput(str, dt, u)
end
export fmiInputDoStepCSOutput

"""
DEPRECATED:

    fmi2InputDoStepCSOutput(comp::FMU2Component, 
                            dt::Real, 
                            u::Array{<:Real})

Sets all FMU inputs to `u`, performs a ´´´fmi2DoStep´´´ and returns all FMU outputs.
"""
function fmi2InputDoStepCSOutput(fmu::FMU2, 
                                 dt::Real, 
                                 u::Array{<:Real})
                                 
    @assert fmi2IsCoSimulation(fmu) ["fmi2InputDoStepCSOutput(...): As in the name, this function only supports CS-FMUs."]

    # fmi2DoStepCS(fmu, dt,
    #              fmu.modelDescription.inputValueReferences,
    #              u,
    #              fmu.modelDescription.outputValueReferences)

    y_refs = fmu.modelDescription.outputValueReferences
    u_refs = fmu.modelDescription.inputValueReferences
    y = zeros(length(y_refs))

    c = fmu.components[end]
    
    y, _ = c(y=y, y_refs=y_refs, u=u, u_refs=u_refs)

    # ignore_derivatives() do
    #     fmi2DoStep(c, dt)
    # end

    return y
end
export fmi2InputDoStepCSOutput

function fmi2DoStepCS(fmu::FMU2, 
    dt::Real,
    setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
    setValues::Array{<:Real} = zeros(Real, 0),
    getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y_refs = setValueReferences
    u_refs = getValueReferences
    y = zeros(length(y_refs))
    u = setValues

    c = fmu.components[end]
    
    y, _ = c(y=y, y_refs=y_refs, u=u, u_refs=u_refs)

    # ignore_derivatives() do
    #     fmi2DoStep(c, dt)
    # end

    return y
end
export fmi2DoStepCS

# FMU wrappers

function fmi2EvaluateME(comp::FMU2Component, args...; kwargs...)
    fmi2EvaluateME(comp.fmu, args...; kwargs...)
end

function fmi2DoStepCS(comp::FMU2Component, args...; kwargs...)
    fmi2DoStepCS(comp.fmu, args...; kwargs...)
end

function fmi2InputDoStepCSOutput(comp::FMU2Component, args...; kwargs...)
    fmi2InputDoStepCSOutput(comp.fmu, args...; kwargs...)
end