#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using DifferentialEquations, DiffEqCallbacks

using ChainRulesCore
import ForwardDiff
using Interpolations: bounds

# helper to collect variable IdSet
function indiciesForValueReferences(fmu::FMU2, 
                                    refs::Array{fmi2ValueReference})
    array = fmi2ValueReference[]
    for i in 1:length(fmu.modelDescription.valueReferences)
        if fmu.modelDescription.valueReferences[i] in refs 
            push!(array, i)
        end
    end
    array
end 

"""
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
        setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
        setValues::Array{<:Real} = zeros(Real, 0), 
        getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0)) 

    _fmi2EvaluateME(fmu, x, t, setValueReferences, setValues, getValueReferences)
end
function fmi2EvaluateME(fmu::FMU2,
    x::Array{<:ForwardDiff.Dual{Tx, Vx, Nx}},
    t,#::Real,
    setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
    setValues::Array{<:Real} = zeros(Real, 0),
    getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0) ) where {Tx, Vx, Nx}

    _fmi2EvaluateME_fd((Tx, Vx, Nx), (Tx, Vx, Nx), fmu, x, t, setValueReferences, setValues, getValueReferences)
end
function fmi2EvaluateME(fmu::FMU2,
    x::Array{<:ForwardDiff.Dual{Tx, Vx, Nx}},
    t,#::Real,
    setValueReferences::Array{fmi2ValueReference},
    setValues::Array{<:ForwardDiff.Dual{Tu, Vu, Nu}},
    getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0) ) where {Tx, Vx, Nx, Tu, Vu, Nu}

    _fmi2EvaluateME_fd((Tx, Vx, Nx), (Tu, Vu, Nu), fmu, x, t, setValueReferences, setValues, getValueReferences)
end

# ForwardDiff backend using the existing ChainRulesCore.frule
# adapted from: https://discourse.julialang.org/t/chainrulescore-and-forwarddiff/61705/8
function _fmi2EvaluateME_fd(TVNx, TVNu, fmu, x, t, setValueReferences, setValues, getValueReferences) 
  
    Tx, Vx, Nx = TVNx
    Tu, Vu, Nu = TVNu

    ȧrgs = [NoTangent(), NoTangent(), collect(ForwardDiff.partials(e) for e in x), ForwardDiff.partials(t), NoTangent(), collect(ForwardDiff.partials(e) for e in setValues), NoTangent()]
    args = [fmi2EvaluateME, fmu, collect(ForwardDiff.value(e) for e in x), ForwardDiff.value(t), setValueReferences, collect(ForwardDiff.value(e) for e in setValues), getValueReferences]

    # ToDo: Find a good fix!
    #ignore_derivatives() do @debug "From $(typeof(args[6]))"
    if typeof(args[6]) == Vector{Any}
        args[6] = convert(Vector{Float64}, args[6])
        #ignore_derivatives do @debug "To $(typeof(args[6]))"
    end 

    ȧrgs = (ȧrgs...,)
    args = (args...,)

    @assert typeof(args[3]) == Vector{Float64} "After conversion, `x` is still an invalid type `$(typeof(args[3]))`."
     
    y, _, dx, _, _, du, _ = ChainRulesCore.frule(ȧrgs, args...)

    if Vx != Float64
        Vx = Float64
    end

    if Vu != Float64
        Vu = Float64
    end

    # original function returns [dx_1, ..., dx_n, y_1, ..., y_m]
    # ToDo: Add sensitivities (partials) from du -> (dx, y)

    [collect( ForwardDiff.Dual{Tx, Vx, Nx}(y[i], dx[i]) for i in 1:length(dx) )...]
end

function _fmi2EvaluateME(fmu::FMU2,
                      x::Array{<:Real},
                      t::Real,
                      setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                      setValues::Array{<:Real} = zeros(Real, 0),
                      getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    
    @assert fmi2IsModelExchange(fmu) ["fmi2EvaluateME(...): As in the name, this function only supports ME-FMUs."]

    setter = (length(setValueReferences) > 0)
    getter = (length(getValueReferences) > 0)

    if setter
        @assert length(setValueReferences) == length(setValues) ["fmi2EvaluateME(...): `setValueReferences` and `setValues` need to be the same length!"]
    end

    comp = fmu.components[end]

    if setter
        fmi2SetReal(comp, setValueReferences, setValues)
    end

    fmi2SetContinuousStates(comp, x)

    if t >= 0.0
        # discrete = (comp.fmu.hasStateEvents || comp.fmu.hasTimeEvents)
        # if ( discrete && comp.state == fmi2ComponentStateEventMode && comp.eventInfo.newDiscreteStatesNeeded == fmi2False) ||
        #    (!discrete && comp.state == fmi2ComponentStateContinuousTimeMode)
        fmi2SetTime(comp, t)
        # end
    end
    
    y = []
    if getter
        y = fmi2GetReal(comp, getValueReferences)
    end

    dx = fmi2GetDerivatives(comp)

    return [dx..., y...]
end

function evaluateJacobians(fmu::FMU2,
                            x::Array{<:Real},
                            t::Real = fmu.components[end].t,
                            setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                            setValues::Array{<:Real} = zeros(Real, 0),
                            getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    comp = fmu.components[end]
    rdx = vcat(fmu.modelDescription.derivativeValueReferences, getValueReferences) 
    # rx = fmu.modelDescription.stateValueReferences
    # ru = setValueReferences
    # comp.jac_ẋy_x = zeros(length(rdx), length(rx))
    # comp.jac_ẋy_u = zeros(length(rdx), length(ru))
    # return nothing

    setter = (length(setValueReferences) > 0)
    getter = (length(getValueReferences) > 0)

    comp = fmu.components[end]

    fmi2SetContinuousStates(comp, x)
    if t >= 0.0
        fmi2SetTime(comp, t)
    end

    stateBefore = comp.state
    if comp.state != fmi2ComponentStateContinuousTimeMode
        fmi2EnterContinuousTimeMode(comp)
    end

    rdx = vcat(fmu.modelDescription.derivativeValueReferences, getValueReferences) 
    rx = fmu.modelDescription.stateValueReferences
    ru = setValueReferences

    if comp.jac_x != x || comp.jac_t != t || comp.jac_u != setValues 

        # Jacobian ∂ ẋy / ∂ x
        if size(comp.jac_ẋy_x) != (length(rdx), length(rx))
            comp.jac_ẋy_x = zeros(length(rdx), length(rx))
        end 
        comp.jacobianUpdate!(comp.jac_ẋy_x, comp, rdx, rx)

        # Jacobian ∂ ẋy / ∂ u
        if setter 
            if size(comp.jac_ẋy_u) != (length(rdx), length(ru))
                comp.jac_ẋy_u = zeros(length(rdx), length(ru))
            end 
            comp.jacobianUpdate!(comp.jac_ẋy_u, comp, rdx, ru)

        end

        comp.jac_u = setValues
        comp.jac_x = x
        comp.jac_t = t

    end

    if comp.state != stateBefore
        fmi2EnterEventMode(comp)
    end

end

function ChainRulesCore.rrule(::typeof(fmi2EvaluateME), 
                              fmu::FMU2,
                              x::Array{<:Real},
                              t::Real = comp.t,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y = fmi2EvaluateME(fmu, x, t, setValueReferences, setValues, getValueReferences)
    
    function fmi2EvaluateME_pullback(ȳ)

        setter = (length(setValueReferences) > 0)
        getter = (length(getValueReferences) > 0)

        if setter
            @assert length(setValueReferences) == length(setValues) ["ChainRulesCore.rrule(fmi2EvaluateME, ...): `setValueReferences` and `setValues` need to be the same length!"]
        end
        
        evaluateJacobians(fmu, x, t, setValueReferences, setValues, getValueReferences)
        comp = fmu.components[end]

        n_dx_x = @thunk(comp.jac_ẋy_x' * ȳ)
        n_dx_u = ZeroTangent()
        
        if setter
            n_dx_u = @thunk(comp.jac_ẋy_u' * ȳ)
        end

        f̄ = NoTangent()
        f̄mu = ZeroTangent()
        x̄ = n_dx_x
        t̄ = ZeroTangent()
        s̄etValueReferences = ZeroTangent() 
        s̄etValues = n_dx_u
        ḡetValueReferences = ZeroTangent()
       
        return f̄, f̄mu, x̄, t̄, s̄etValueReferences, s̄etValues, ḡetValueReferences
    end
    return y, fmi2EvaluateME_pullback
end

function ChainRulesCore.frule((Δself, Δcomp, Δx, Δt, ΔsetValueReferences, ΔsetValues, ΔgetValueReferences), 
                              ::typeof(fmi2EvaluateME), 
                              fmu, #::FMU2Component,
                              x,#::Array{<:Real},
                              t,#::Real = comp.t,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y = fmi2EvaluateME(fmu, x, t, setValueReferences, setValues, getValueReferences)

    function fmi2EvaluateME_pullforward(Δx, ΔsetValues)
        setter = (length(setValueReferences) > 0)
        getter = (length(getValueReferences) > 0)

        if setter
            @assert length(setValueReferences) == length(setValues) ["ChainRulesCore.frule(fmi2EvaluateME, ...): `setValueReferences` and `setValues` need to be the same length!"]
        end

        comp = fmu.components[end]

        # OLD Code START
        evaluateJacobians(fmu, x, t, setValueReferences, setValues, getValueReferences)
        
        n_dx_x = comp.jac_ẋy_x * Δx
        n_dx_u = ZeroTangent()  
        if setter
            n_dx_u = comp.jac_ẋy_u * ΔsetValues
        end
        # OLD Code END
        
        # TEST START
        # global Δx, rx, rdx
        # n_dx_x = NoTangent()
        # n_dx_u = NoTangent()
        # rdx = vcat(comp.fmu.modelDescription.derivativeValueReferences, getValueReferences) 
        # rx = comp.fmu.modelDescription.stateValueReferences
        # ru = setValueReferences
        # n_dx_x = fmi2GetDirectionalDerivative(comp, rdx, rx, Δx)
        # if setter
        #     n_dx_u = fmi2GetDirectionalDerivative(comp, rdx, ru, ΔsetValues)
        # end
        # TEST END
        

        f̄mu = ZeroTangent()
        x̄ = n_dx_x 
        t̄ = ZeroTangent()
        s̄etValueReferences = ZeroTangent()
        s̄etValues = n_dx_u
        ḡetValueReferences = ZeroTangent()
       
        return (f̄mu, x̄, t̄, s̄etValueReferences, s̄etValues, ḡetValueReferences)
    end
    return (y, fmi2EvaluateME_pullforward(Δx, ΔsetValues)...)
end

"""
Performs a fmiDoStep for CS-FMUs (note, that fmiDoStep is for CS-FMUs only).

Optional, FMU-values can be set via keyword arguments `setValueReferences` and `setValues`.
Optional, FMU-values can be retrieved by keyword argument `getValueReferences`.

Function returns the FMU-values for the optional keyword argument `getValueReferences`.
The CS-FMU performs one macro step with step size `dt`. Dependent on the integrated numerical solver, the FMU may perform multiple (internal) micro steps if needed to meet solver requirements (stability/accuracy). These micro steps are hidden by FMI2.
"""
function fmi2DoStepCS(fmu::FMU2, 
                      dt::Real,
                      setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                      setValues::Array{<:Real} = zeros(Real, 0),
                      getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    _fmi2DoStepCS(fmu, dt, setValueReferences, setValues, getValueReferences)
end
function fmi2DoStepCS(fmu::FMU2,
    dt,#::Real,
    setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
    setValues::Array{<:ForwardDiff.Dual{Tu, Vu, Nu}} = Array{ForwardDiff.Dual{Tu, Vu, Nu}}(),
    getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0) ) where {Tu, Vu, Nu}

    _fmi2DoStepCS_fd((Td, Vd, Nd), fmu, dt, setValueReferences, setValues, getValueReferences)
end

# Helper because keyword arguments are (currently) not supported by Zygote.
function _fmi2DoStepCS(fmu::FMU2, 
                       dt::Real, 
                       setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                       setValues::Array{<:Real} = zeros(Real, 0),
                       getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    
    comp = fmu.components[end]
    @assert fmi2IsCoSimulation(comp.fmu) ["fmi2DoStepCS(...): As in the name, this function only supports CS-FMUs."]
    @assert length(setValueReferences) == length(setValues) ["fmi2DoStepCS(...): `setValueReferences` ($(length(setValueReferences))) and `setValues` ($(length(setValues))) need to be the same length!"]

    if length(setValueReferences) > 0
        fmi2SetReal(comp, setValueReferences, setValues)
    end

    fmi2DoStep(fmu, dt)

    y = zeros(Float64, 0)

    if length(getValueReferences) > 0
        y = fmi2GetReal(comp, getValueReferences)
    end

    y
end

# ForwardDiff backend using the existing ChainRulesCore.frule
# adapted from: https://discourse.julialang.org/t/chainrulescore-and-forwarddiff/61705/8
function _fmi2DoStepCS_fd(TVNu, 
                          fmu,
                          dt, 
                          setValueReferences, 
                          setValues, 
                          getValueReferences) 
  
    Tu, Vu, Nu = TVNu
    comp = fmu.components[end]

    ȧrgs = [NoTangent(), NoTangent(), ForwardDiff.partials(dt), NoTangent(), collect(ForwardDiff.partials(e) for e in setValues), NoTangent()]
    args = [fmi2DoStepCS, fmu, ForwardDiff.value(dt), setValueReferences, collect(ForwardDiff.value(e) for e in setValues), getValueReferences]

    # ToDo: Find a good fix!
    if typeof(args[5]) == Vector{Any}
        args[5] = convert(Vector{Float64}, args[5])
    end 

    ȧrgs = (ȧrgs...,)
    args = (args...,)
     
    y, _, _, _, du, _ = ChainRulesCore.frule(ȧrgs, args...)

    if Vu != Float64
        Vu = Float64
    end

    # original function returns [y_1, ..., y_m]
    [collect( ForwardDiff.Dual{Tu, Vu, Nu}(y[i], du[i]) for i in 1:length(du) )...]
end

function ChainRulesCore.rrule(::typeof(fmi2DoStepCS), 
                              fmu::FMU2,
                              dt::Real = comp.t,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    comp = fmu.components[end]
    y = fmi2DoStepCS(comp, dt, setValueReferences, setValues, getValueReferences)
    function fmi2DoStepCS_pullback(ȳ)
        setter = (length(setValueReferences) > 0)
        getter = (length(getValueReferences) > 0)

        if setter
            @assert length(setValueReferences) == length(setValues) ["ChainRulesCore.rrule(fmi2DoStepCS, ...): `setValueReferences` and `setValues` need to be the same length!"]
        end

        rdx = getValueReferences
        ru = setValueReferences

        n_dx_u = ZeroTangent()

        if getter
            if size(comp.jac_ẋy_u) != (length(rdx), length(ru))
                comp.jac_ẋy_u = zeros(length(rdx), length(ru))
            end 
            comp.jacobianUpdate!(comp.jac_ẋy_u, comp, rdx, ru)

            n_dx_u = @thunk(comp.jac_ẋy_u' * ȳ)
        end

        f̄ = NoTangent()
        f̄mu = ZeroTangent()
        d̄t = ZeroTangent()
        s̄etValueReferences = ZeroTangent()
        s̄etValues = n_dx_u
        ḡetValueReferences = ZeroTangent()

        return (f̄, f̄mu, d̄t, s̄etValueReferences, s̄etValues, ḡetValueReferences)
    end
    return y, fmi2DoStepCS_pullback
end

function ChainRulesCore.frule((Δself, Δcomp, Δdt, ΔsetValueReferences, ΔsetValues, ΔgetValueReferences), 
                              ::typeof(fmi2DoStepCS), 
                              fmu, #::FMU2,
                              dt,#::Real = comp.t,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    comp = fmu.components[end]
    y = fmi2DoStepCS(fmu, dt, setValueReferences, setValues, getValueReferences)
    function fmi2DoStepCS_pullforward(ΔsetValues)
        setter = (length(setValueReferences) > 0)
        getter = (length(getValueReferences) > 0)

        if setter
            @assert length(setValueReferences) == length(setValues) ["ChainRulesCore.frule(fmi2DoStepCS, ...): `setValueReferences` and `setValues` need to be the same length!"]
        end

        rdx = getValueReferences
        ru = setValueReferences

        n_dx_u = ZeroTangent()

        if getter
            if size(comp.jac_ẋy_u) != (length(rdx), length(ru))
                comp.jac_ẋy_u = zeros(length(rdx), length(ru))
            end 
            comp.jacobianUpdate!(comp.jac_ẋy_u, comp, rdx, ru)

            n_dx_u = comp.jac_ẋy_u * ΔsetValues
        end

        f̄mu = ZeroTangent()
        d̄t = ZeroTangent()
        s̄etValueReferences = ZeroTangent()
        s̄etValues = n_dx_u
        ḡetValueReferences = ZeroTangent()

        return (f̄mu, d̄t, s̄etValueReferences, s̄etValues, ḡetValueReferences)
    end
    return (y, fmi2DoStepCS_pullforward(ΔsetValues)...)
end

"""
    fmi2InputDoStepCSOutput(comp::FMU2Component, 
                            dt::Real, 
                            u::Array{<:Real})

Sets all FMU inputs to `u`, performs a ´´´fmi2DoStep´´´ and returns all FMU outputs.
"""
function fmi2InputDoStepCSOutput(fmu::FMU2, 
                                 dt::Real, 
                                 u::Array{<:Real})
                                 
    @assert fmi2IsCoSimulation(fmu) ["fmi2InputDoStepCSOutput(...): As in the name, this function only supports CS-FMUs."]

    fmi2DoStepCS(fmu, dt,
                 fmu.modelDescription.inputValueReferences,
                 u,
                 fmu.modelDescription.outputValueReferences)
end

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