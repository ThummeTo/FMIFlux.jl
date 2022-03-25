#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using DifferentialEquations, DiffEqCallbacks

using ChainRulesCore
import ForwardDiff, Zygote

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
function fmi2EvaluateME(comp::FMU2Component,
        x::Array{<:Real},
        t = comp.t,#::Real,
        setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
        setValues::Array{<:Real} = zeros(Real, 0), 
        getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0)) 

    _fmi2EvaluateME(comp, x, t, setValueReferences, setValues, getValueReferences)
end
function fmi2EvaluateME(comp::FMU2Component,
    x::Array{<:ForwardDiff.Dual{Tx, Vx, Nx}},
    t = comp.t,#::Real,
    setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
    setValues::Array{<:Real} = zeros(Real, 0),
    getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0) ) where {Tx, Vx, Nx}

    _fmi2EvaluateME_fd((Tx, Vx, Nx), (Tx, Vx, Nx), comp, x, t, setValueReferences, setValues, getValueReferences)
end
function fmi2EvaluateME(comp::FMU2Component,
    x::Array{<:ForwardDiff.Dual{Tx, Vx, Nx}},
    t,#::Real,
    setValueReferences::Array{fmi2ValueReference},
    setValues::Array{<:ForwardDiff.Dual{Tu, Vu, Nu}},
    getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0) ) where {Tx, Vx, Nx, Tu, Vu, Nu}

    _fmi2EvaluateME_fd((Tx, Vx, Nx), (Tu, Vu, Nu), comp, x, t, setValueReferences, setValues, getValueReferences)
end

# ForwardDiff backend using the existing ChainRulesCore.frule
# adapted from: https://discourse.julialang.org/t/chainrulescore-and-forwarddiff/61705/8
function _fmi2EvaluateME_fd(TVNx, TVNu, comp, x, t, setValueReferences, setValues, getValueReferences) 
  
    Tx, Vx, Nx = TVNx
    Tu, Vu, Nu = TVNu

    ȧrgs = [NoTangent(), NoTangent(), collect(ForwardDiff.partials(e) for e in x), ForwardDiff.partials(t), NoTangent(), collect(ForwardDiff.partials(e) for e in setValues), NoTangent()]
    args = [fmi2EvaluateME, comp, collect(ForwardDiff.value(e) for e in x), ForwardDiff.value(t), setValueReferences, collect(ForwardDiff.value(e) for e in setValues), getValueReferences]

    # ToDo: Find a good fix!
    #ignore_derivatives() do @debug "From $(typeof(args[6]))"
    if typeof(args[6]) == Vector{Any}
        args[6] = convert(Vector{Float64}, args[6])
        #ignore_derivatives do @debug "To $(typeof(args[6]))"
    end 

    ȧrgs = (ȧrgs...,)
    args = (args...,)

    @assert (typeof(args[3]) == Vector{Float64}) || (typeof(args[3]) == Vector{Float32}) "After conversion, `x` is still an invalid type `$(typeof(args[3]))`."
     
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

function _fmi2EvaluateME(comp::FMU2Component,
                      x::Array{<:Real},
                      t::Real,
                      setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                      setValues::Array{<:Real} = zeros(Real, 0),
                      getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    
    @assert fmi2IsModelExchange(comp.fmu) ["fmi2EvaluateME(...): As in the name, this function only supports ME-FMUs."]

    setter = (length(setValueReferences) > 0)
    getter = (length(getValueReferences) > 0)

    if setter
        @assert length(setValueReferences) == length(setValues) ["fmi2EvaluateME(...): `setValueReferences` and `setValues` need to be the same length!"]
    end

    if t >= 0.0 
        fmi2SetTime(comp, t)
    end

    if setter
        fmi2SetReal(comp, setValueReferences, setValues)
    end

    fmi2SetContinuousStates(comp, x)
    
    y = []
    if getter
        y = fmi2GetReal(comp, getValueReferences)
    end

    dx = fmi2GetDerivatives(comp)

    [dx..., y...] 
end

function ChainRulesCore.rrule(::typeof(fmi2EvaluateME), 
                              comp::FMU2Component,
                              x::Array{<:Real},
                              t::Real = comp.t,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y = fmi2EvaluateME(comp, x, t, setValueReferences, setValues, getValueReferences)
    if comp.fmu.ẋ_interp !== nothing
        y = comp.fmu.ẋ_interp(t)
    end
    
    function fmi2EvaluateME_pullback(ȳ)
        setter = (length(setValueReferences) > 0)
        getter = (length(getValueReferences) > 0)

        if setter
            @assert length(setValueReferences) == length(setValues) ["ChainRulesCore.rrule(fmi2EvaluateME, ...): `setValueReferences` and `setValues` need to be the same length!"]
        end

        if t >= 0.0
            fmi2SetTime(comp, t)
        end

        fmi2SetContinuousStates(comp, x)

        rdx = vcat(comp.fmu.modelDescription.derivativeValueReferences, getValueReferences) 
        rx = comp.fmu.modelDescription.stateValueReferences
        ru = setValueReferences

        n_dx_x = NoTangent()
        n_dx_u = NoTangent()

        if comp.senseFunc == :full || comp.senseFunc == :directionalDerivatives || comp.senseFunc == :auto
            # OPTIMIZATION: compute new jacobians only if system state or time changed, otherwise return the cached one
            if comp.jac_x != x || comp.jac_t != t 

                if size(comp.jac_dxy_x) != (length(rdx), length(rx))
                    comp.jac_dxy_x = zeros(length(rdx), length(rx))
                end 
                comp.jacobianFct(comp.jac_dxy_x, comp, rdx, rx)

                if size(comp.jac_dxy_u) != (length(rdx), length(ru))
                    comp.jac_dxy_u = zeros(length(rdx), length(ru))
                end
                comp.jacobianFct(comp.jac_dxy_u, comp, rdx, ru)

                comp.jac_x = x
                comp.jac_t = t
            end

            n_dx_x = @thunk(comp.jac_dxy_x' * ȳ)
            
            if setter
                n_dx_u = @thunk(comp.jac_dxy_u' * ȳ)
            end
        elseif comp.senseFunc == :adjointDerivatives  
            @assert false "Adjoint Derivatives not supported by FMI2."
        else
            @assert false "`senseFunc=$(comp.senseFunc)` unknown value for `senseFunc`."
        end

        f̄ = NoTangent()
        c̄omp = ZeroTangent()
        x̄ = n_dx_x
        t̄ = ZeroTangent()
        s̄etValueReferences = ZeroTangent() 
        s̄etValues = n_dx_u
        ḡetValueReferences = ZeroTangent()
       
        return f̄, c̄omp, x̄, t̄, s̄etValueReferences, s̄etValues, ḡetValueReferences
    end
    return y, fmi2EvaluateME_pullback
end

function ChainRulesCore.frule((Δself, Δcomp, Δx, Δt, ΔsetValueReferences, ΔsetValues, ΔgetValueReferences), 
                              ::typeof(fmi2EvaluateME), 
                              comp, #::FMU2Component,
                              x,#::Array{<:Real},
                              t,#::Real = comp.t,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y = fmi2EvaluateME(comp, x, t, setValueReferences, setValues, getValueReferences)
    if comp.fmu.ẋ_interp !== nothing 
        y = comp.fmu.ẋ_interp(t)
    end

    function fmi2EvaluateME_pullforward(Δx, ΔsetValues)
        setter = (length(setValueReferences) > 0)
        getter = (length(getValueReferences) > 0)

        if setter
            @assert length(setValueReferences) == length(setValues) ["ChainRulesCore.frule(fmi2EvaluateME, ...): `setValueReferences` and `setValues` need to be the same length!"]
        end

        # already set!
        # if t >= 0.0 
        #     fmi2SetTime(comp, t)
        # end
        
        # if all(isa.(x, ForwardDiff.Dual))
        #     xf = collect(ForwardDiff.value(e) for e in x)
        #     fmi2SetContinuousStates(comp, xf)
        # else
        #     fmi2SetContinuousStates(comp, x)
        # end
       
        rdx = vcat(comp.fmu.modelDescription.derivativeValueReferences, getValueReferences) 
        rx = comp.fmu.modelDescription.stateValueReferences
        ru = setValueReferences

        n_dx_x = NoTangent()
        n_dx_u = NoTangent()

        if comp.senseFunc == :auto || comp.senseFunc == :full
            # OPTIMIZATION: compute new jacobians only if system state or time changed, otherwise return the cached one
            # ToDo: Optimize for getDirectionalDerivatives with seed vector Δx
            if comp.jac_x != x || comp.jac_t != t 

                if size(comp.jac_dxy_x) != (length(rdx), length(rx))
                    comp.jac_dxy_x = zeros(length(rdx), length(rx))
                end
                comp.jacobianFct(comp.jac_dxy_x, comp, rdx, rx)

                if size(comp.jac_dxy_u) != (length(rdx), length(ru))
                    comp.jac_dxy_u = zeros(length(rdx), length(ru))
                end
                comp.jacobianFct(comp.jac_dxy_u, comp, rdx, ru)
               
                comp.jac_x = x
                comp.jac_t = t
            end

            n_dx_x = comp.jac_dxy_x * Δx
            
            if setter
                n_dx_u = comp.jac_dxy_u * ΔsetValues
            end
        elseif comp.senseFunc == :directionalDerivatives
            n_dx_x = fmi2GetDirectionalDerivative(comp, rdx, rx, Δx)
            if setter
                n_dx_u = fmi2GetDirectionalDerivative(comp, rdx, ru, ΔsetValues)
            end
        elseif comp.senseFunc == :adjointDerivatives  
            @assert false "Adjoint Derivatives not supported by FMI2."
        else
            @assert false "`senseFunc=$(comp.senseFunc)` unknown value for `senseFunc`."
        end

        c̄omp = ZeroTangent()
        x̄ = n_dx_x 
        t̄ = ZeroTangent()
        s̄etValueReferences = ZeroTangent()
        s̄etValues = n_dx_u
        ḡetValueReferences = ZeroTangent()
       
        return (c̄omp, x̄, t̄, s̄etValueReferences, s̄etValues, ḡetValueReferences)
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
function fmi2DoStepCS(comp::FMU2Component, 
                      dt::Real,
                      setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                      setValues::Array{<:Real} = zeros(Real, 0),
                      getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    _fmi2DoStepCS(comp, dt, setValueReferences, setValues, getValueReferences)
end
function fmi2DoStepCS(comp::FMU2Component,
    dt,#::Real,
    setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
    setValues::Array{<:ForwardDiff.Dual{Tu, Vu, Nu}} = Array{ForwardDiff.Dual{Tu, Vu, Nu}}(),
    getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0) ) where {Tu, Vu, Nu}

    _fmi2DoStepCS_fd((Td, Vd, Nd), comp, dt, setValueReferences, setValues, getValueReferences)
end

# Helper because keyword arguments are (currently) not supported by Zygote.
function _fmi2DoStepCS(comp::FMU2Component, 
                       dt::Real, 
                       setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                       setValues::Array{<:Real} = zeros(Real, 0),
                       getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    
    @assert fmi2IsCoSimulation(comp.fmu) ["fmi2DoStepCS(...): As in the name, this function only supports CS-FMUs."]
    @assert length(setValueReferences) == length(setValues) ["fmi2DoStepCS(...): `setValueReferences` ($(length(setValueReferences))) and `setValues` ($(length(setValues))) need to be the same length!"]

    if length(setValueReferences) > 0
        fmi2SetReal(comp, setValueReferences, setValues)
    end

    fmi2DoStep(comp, dt)

    y = zeros(Float64, 0)

    if length(getValueReferences) > 0
        y = fmi2GetReal(comp, getValueReferences)
    end

    y
end

# ForwardDiff backend using the existing ChainRulesCore.frule
# adapted from: https://discourse.julialang.org/t/chainrulescore-and-forwarddiff/61705/8
function _fmi2DoStepCS_fd(TVNu, 
    comp, 
                          dt, 
                          setValueReferences, 
                          setValues, 
                          getValueReferences) 
  
    Tu, Vu, Nu = TVNu

    ȧrgs = [NoTangent(), NoTangent(), ForwardDiff.partials(dt), NoTangent(), collect(ForwardDiff.partials(e) for e in setValues), NoTangent()]
    args = [fmi2DoStepCS, comp, ForwardDiff.value(dt), setValueReferences, collect(ForwardDiff.value(e) for e in setValues), getValueReferences]

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
                              comp::FMU2Component,
                              dt::Real = comp.t,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y = fmi2DoStepCS(comp, dt, setValueReferences, setValues, getValueReferences)
    function fmi2DoStepCS_pullback(ȳ)
        setter = (length(setValueReferences) > 0)
        getter = (length(getValueReferences) > 0)

        if setter
            @assert length(setValueReferences) == length(setValues) ["ChainRulesCore.rrule(fmi2DoStepCS, ...): `setValueReferences` and `setValues` need to be the same length!"]
        end

        rdx = getValueReferences
        rx = setValueReferences

        n_dx_u = ZeroTangent()

        if getter
            mat = zeros(length(rdx), length(rx))
            comp.jacobianFct(mat, comp, rdx, rx)
            n_dx_u = @thunk(mat' * ȳ)
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
                              comp, #::FMU2,
                              dt,#::Real = comp.t,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y = fmi2DoStepCS(comp, dt, setValueReferences, setValues, getValueReferences)
    function fmi2DoStepCS_pullforward(ΔsetValues)
        setter = (length(setValueReferences) > 0)
        getter = (length(getValueReferences) > 0)

        if setter
            @assert length(setValueReferences) == length(setValues) ["ChainRulesCore.frule(fmi2DoStepCS, ...): `setValueReferences` and `setValues` need to be the same length!"]
        end

        rdx = getValueReferences
        rx = setValueReferences

        n_dx_u = ZeroTangent()

        if getter
            mat = zeros(length(rdx), length(rx))
            comp.jacobianFct(mat, comp, rdx, rx)
            n_dx_u = mat * ΔsetValues
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
function fmi2InputDoStepCSOutput(comp::FMU2Component, 
                                 dt::Real, 
                                 u::Array{<:Real})
                                 
    @assert fmi2IsCoSimulation(comp.fmu) ["fmi2InputDoStepCSOutput(...): As in the name, this function only supports CS-FMUs."]

    fmi2DoStepCS(comp, dt,
                 comp.fmu.modelDescription.inputValueReferences,
                 u,
                 comp.fmu.modelDescription.outputValueReferences)
end

# FMU wrappers

function fmi2EvaluateME(fmu::FMU2, args...; kwargs...)
    fmi2EvaluateME(fmu.components[end], args...; kwargs...)
end

function fmi2DoStepCS(fmu::FMU2, args...; kwargs...)
    fmi2DoStepCS(fmu.components[end], args...; kwargs...)
end

function fmi2InputDoStepCSOutput(fmu::FMU2, args...; kwargs...)
    fmi2InputDoStepCSOutput(fmu.components[end], args...; kwargs...)
end