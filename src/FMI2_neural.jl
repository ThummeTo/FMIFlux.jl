#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using DifferentialEquations, DiffEqCallbacks

using ChainRulesCore
import ForwardDiff

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
Builds the jacobian over the FMU `fmu` for FMU value references `rdx` and `rx`, so that the function returns the jacobian ∂rdx / ∂rx.

If FMI built-in directional derivatives are supported, they are used.
As fallback, directional derivatives will be sampled with central differences.
For optimization, if the FMU's model description has the optional entry 'dependencies', only dependent variables are sampled/retrieved. This drastically boosts performance for systems with large variable count (like CFD). 

If sampling is used, sampling step size can be set (for each direction individually) using optional argument `steps`.
"""
function fmi2GetJacobian(fmu::FMU2, 
                         rdx::Array{fmi2ValueReference}, 
                         rx::Array{fmi2ValueReference}; 
                         steps::Array{fmi2Real} = ones(fmi2Real, length(rdx)).*1e-5)
    mat = zeros(fmi2Real, length(rdx), length(rx))
    fmi2GetJacobian!(mat, fmu, rdx, rx; steps=steps)
    return mat
end

"""
Fills the jacobian over the FMU `fmu` for FMU value references `rdx` and `rx`, so that the function returns the jacobian ∂rdx / ∂rx.

If FMI built-in directional derivatives are supported, they are used.
As fallback, directional derivatives will be sampled with central differences.
For optimization, if the FMU's model description has the optional entry 'dependencies', only dependent variables are sampled/retrieved. This drastically boosts performance for systems with large variable count (like CFD). 

If sampling is used, sampling step size can be set (for each direction individually) using optional argument `steps`.
"""
function fmi2GetJacobian!(jac::Matrix{fmi2Real}, 
                          fmu::FMU2, 
                          rdx::Array{fmi2ValueReference}, 
                          rx::Array{fmi2ValueReference}; 
                          steps::Array{fmi2Real} = ones(fmi2Real, length(rdx)).*1e-5)

    @assert size(jac) == (length(rdx), length(rx)) ["fmi2GetJacobian!: Dimension missmatch between `jac` $(size(jac)), `rdx` ($length(rdx)) and `rx` ($length(rx))."]

    if length(rdx) == 0 || length(rx) == 0
        jac = zeros(length(rdx), length(rx))
        return nothing
    end 

    ddsupported = fmi2ProvidesDirectionalDerivative(fmu)

    # ToDo: Pick entries based on dependency matrix!
    #depMtx = fmi2GetDependencies(fmu)
    rdx_inds = collect(fmu.modelDescription.valueReferenceIndicies[vr] for vr in rdx)
    rx_inds  = collect(fmu.modelDescription.valueReferenceIndicies[vr] for vr in rx)
    
    for i in 1:length(rx)

        sensitive_rdx_inds = 1:length(rdx)
        sensitive_rdx = rdx

        # sensitive_rdx_inds = Int64[]
        # sensitive_rdx = fmi2ValueReference[]

        # for j in 1:length(rdx)
        #     if depMtx[rdx_inds[j], rx_inds[i]] != fmi2DependencyIndependent
        #         push!(sensitive_rdx_inds, j)
        #         push!(sensitive_rdx, rdx[j])
        #     end
        # end

        if length(sensitive_rdx) > 0
            if ddsupported
                # doesn't work because indexed-views can`t be passed by reference (to ccalls)
                # fmi2GetDirectionalDerivative!(fmu, sensitive_rdx, [rx[i]], view(jac, sensitive_rdx_inds, i))
                jac[sensitive_rdx_inds, i] = fmi2GetDirectionalDerivative(fmu, sensitive_rdx, [rx[i]])
            else 
                # doesn't work because indexed-views can`t be passed by reference (to ccalls)
                # fmi2SampleDirectionalDerivative!(fmu, sensitive_rdx, [rx[i]], steps, view(jac, sensitive_rdx_inds, i))
                jac[sensitive_rdx_inds, i] = fmi2SampleDirectionalDerivative(fmu, sensitive_rdx, [rx[i]], steps)
            end
        end
    end
     
    return nothing
end

"""
Builds the jacobian over the FMU `fmu` for FMU value references `rdx` and `rx`, so that the function returns the jacobian ∂rdx / ∂rx.

If FMI built-in directional derivatives are supported, they are used.
As fallback, directional derivatives will be sampled with central differences.
No performance optimization, for an optimized version use `fmi2GetJacobian`.

If sampling is used, sampling step size can be set (for each direction individually) using optional argument `steps`.
"""
function fmi2GetFullJacobian(fmu::FMU2, 
                             rdx::Array{fmi2ValueReference}, 
                             rx::Array{fmi2ValueReference}; 
                             steps::Array{fmi2Real} = ones(fmi2Real, length(rdx)).*1e-5)
    mat = zeros(fmi2Real, length(rdx), length(rx))
    fmi2GetFullJacobian!(mat, fmu, rdx, rx; steps=steps)
    return mat
end

"""
Fills the jacobian over the FMU `fmu` for FMU value references `rdx` and `rx`, so that the function returns the jacobian ∂rdx / ∂rx.

If FMI built-in directional derivatives are supported, they are used.
As fallback, directional derivatives will be sampled with central differences.
No performance optimization, for an optimized version use `fmi2GetJacobian!`.

If sampling is used, sampling step size can be set (for each direction individually) using optional argument `steps`.
"""
function fmi2GetFullJacobian!(jac::Matrix{fmi2Real}, 
                              fmu::FMU2, 
                              rdx::Array{fmi2ValueReference}, 
                              rx::Array{fmi2ValueReference}; 
                              steps::Array{fmi2Real} = ones(fmi2Real, length(rdx)).*1e-5)
    @assert size(jac) == (length(rdx),length(rx)) "fmi2GetFullJacobian!: Dimension missmatch between `jac` $(size(jac)), `rdx` ($length(rdx)) and `rx` ($length(rx))."

    @warn "`fmi2GetFullJacobian!` is for benchmarking only, please use `fmi2GetJacobian`."

    if length(rdx) == 0 || length(rx) == 0
        jac = zeros(length(rdx), length(rx))
        return nothing
    end 

    if fmi2ProvidesDirectionalDerivative(fmu)
        for i in 1:length(rx)
            jac[:,i] = fmi2GetDirectionalDerivative(fmu, rdx, [rx[i]])
        end
    else
        jac = fmi2SampleDirectionalDerivative(fmu, rdx, rx)
    end

    return nothing
end

"""
Performs something similar to `fmiDoStep` for ME-FMUs (note, that fmiDoStep is for CS-FMUs only).
Event handling (state- and time-events) is supported. If you don't want events to be handled, you can disable event-handling for the NeuralFMU `nfmu` with the attribute `eventHandling = false`.

Optional, additional FMU-values can be set via keyword arguments `setValueReferences` and `setValues`.
Optional, additional FMU-values can be retrieved by keyword argument `getValueReferences`.

Function takes the current system state array ("x") and returns an array with state derivatives ("x dot") and optionally the FMU-values for `getValueReferences`.
Setting the FMU time via argument `t` is optional, if not set, the current time of the ODE solver around the NeuralFMU is used.
"""
function fmi2DoStepME(fmu::FMU2,
        x::Array{<:Real},
        t = -1.0,#::Real,
        setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
        setValues::Array{<:Real} = zeros(Real, 0), 
        getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0)) 

    _fmi2DoStepME(fmu, x, t, setValueReferences, setValues, getValueReferences)
end
function fmi2DoStepME(fmu::FMU2,
    x::Array{<:ForwardDiff.Dual{Tx, Vx, Nx}},
    t = -1.0,#::Real,
    setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
    setValues::Array{<:Real} = zeros(Real, 0),
    getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0) ) where {Tx, Vx, Nx}

    _fmi2DoStepME_fd((Tx, Vx, Nx), (Tx, Vx, Nx), fmu, x, t, setValueReferences, setValues, getValueReferences)
end
function fmi2DoStepME(fmu::FMU2,
    x::Array{<:ForwardDiff.Dual{Tx, Vx, Nx}},
    t,#::Real,
    setValueReferences::Array{fmi2ValueReference},
    setValues::Array{<:ForwardDiff.Dual{Tu, Vu, Nu}},
    getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0) ) where {Tx, Vx, Nx, Tu, Vu, Nu}

    _fmi2DoStepME_fd((Tx, Vx, Nx), (Tu, Vu, Nu), fmu, x, t, setValueReferences, setValues, getValueReferences)
end

# ForwardDiff backend using the existing ChainRulesCore.frule
# adapted from: https://discourse.julialang.org/t/chainrulescore-and-forwarddiff/61705/8
function _fmi2DoStepME_fd(TVNx, TVNu, fmu, x, t, setValueReferences, setValues, getValueReferences) #where {T, V, N} #where {T} # <:ForwardDiff.Dual}
  
    Tx, Vx, Nx = TVNx
    Tu, Vu, Nu = TVNu

    ȧrgs = [NoTangent(), NoTangent(), collect(ForwardDiff.partials(e) for e in x), ForwardDiff.partials(t), NoTangent(), collect(ForwardDiff.partials(e) for e in setValues), NoTangent()]
    args = [fmi2DoStepME, fmu, collect(ForwardDiff.value(e) for e in x), ForwardDiff.value(t), setValueReferences, collect(ForwardDiff.value(e) for e in setValues), getValueReferences]

    # ToDo: Find a good fix!
    if typeof(args[6]) == Vector{Any}
        args[6] = convert(Vector{Float64}, args[6])
    end 

    ȧrgs = (ȧrgs...,)
    args = (args...,)
     
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

function _fmi2DoStepME(fmu::FMU2,
                      x::Array{<:Real},
                      t::Real,
                      setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                      setValues::Array{<:Real} = zeros(Real, 0),
                      getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    
    @assert fmi2IsModelExchange(fmu) ["fmi2DoStepME(...): As in the name, this function only supports ME-FMUs."]

    setter = (length(setValueReferences) > 0)
    getter = (length(getValueReferences) > 0)

    if setter
        @assert length(setValueReferences) == length(setValues) ["fmi2DoStepME(...): `setValueReferences` and `setValues` need to be the same length!"]
    end

    if t >= 0.0 
        fmi2SetTime(fmu, t)
    end

    if setter
        fmi2SetReal(fmu, setValueReferences, setValues)
    end

    fmi2SetContinuousStates(fmu, x)
    
    y = []
    if getter
        y = fmi2GetReal(fmu, getValueReferences)
    end

    dx = fmi2GetDerivatives(fmu)

    [dx..., y...] 
end

function ChainRulesCore.rrule(::typeof(fmi2DoStepME), 
                              fmu::FMU2,
                              x::Array{<:Real},
                              t::Real = -1.0,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y = fmi2DoStepME(fmu, x, t, setValueReferences, setValues, getValueReferences)
    function fmi2DoStepME_pullback(ȳ)
        setter = (length(setValueReferences) > 0)
        getter = (length(getValueReferences) > 0)

        if setter
            @assert length(setValueReferences) == length(setValues) ["ChainRulesCore.rrule(fmi2DoStepME, ...): `setValueReferences` and `setValues` need to be the same length!"]
        end

        if t >= 0.0
            fmi2SetTime(fmu, t)
        end

        fmi2SetContinuousStates(fmu, x)

        rdx = vcat(fmu.modelDescription.derivativeValueReferences, getValueReferences) 
        rx = fmu.modelDescription.stateValueReferences
        ru = setValueReferences

        # OPTIMIZATION: compute new jacobians only if system state or time changed, otherwise return the cached one
        if fmu.jac_x != x || fmu.jac_t != t 

            if size(fmu.jac_dxy_x) != (length(rdx), length(rx))
                fmu.jac_dxy_x = fmi2GetJacobian(fmu, rdx, rx)
            else 
                fmi2GetJacobian!(fmu.jac_dxy_x, fmu, rdx, rx)
            end

            if size(fmu.jac_dxy_u) != (length(rdx), length(ru))
                fmu.jac_dxy_u = fmi2GetJacobian(fmu, rdx, ru)
            else 
                fmi2GetJacobian!(fmu.jac_dxy_u, fmu, rdx, ru)
            end

            fmu.jac_x = x
            fmu.jac_t = t
        end

        n_dx_x = @thunk(fmu.jac_dxy_x' * ȳ)
        n_dx_u = NoTangent()

        if setter
            n_dx_u = @thunk(fmu.jac_dxy_u' * ȳ)
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
    return y, fmi2DoStepME_pullback
end

function ChainRulesCore.frule((Δself, Δfmu, Δx, Δt, ΔsetValueReferences, ΔsetValues, ΔgetValueReferences), 
                              ::typeof(fmi2DoStepME), 
                              fmu, #::FMU2,
                              x,#::Array{<:Real},
                              t,#::Real = -1.0,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y = fmi2DoStepME(fmu, x, t, setValueReferences, setValues, getValueReferences)
    function fmi2DoStepME_pullforward(Δx, ΔsetValues)
        setter = (length(setValueReferences) > 0)
        getter = (length(getValueReferences) > 0)

        if setter
            @assert length(setValueReferences) == length(setValues) ["ChainRulesCore.frule(fmi2DoStepME, ...): `setValueReferences` and `setValues` need to be the same length!"]
        end

        if t >= 0.0 
            fmi2SetTime(fmu, t)
        end
        
        if all(isa.(x, ForwardDiff.Dual))
            xf = collect(ForwardDiff.value(e) for e in x)
            fmi2SetContinuousStates(fmu, xf)
        else
            fmi2SetContinuousStates(fmu, x)
        end
       
        rdx = vcat(fmu.modelDescription.derivativeValueReferences, getValueReferences) 
        rx = fmu.modelDescription.stateValueReferences
        ru = setValueReferences

        # OPTIMIZATION: compute new jacobians only if system state or time changed, otherwise return the cached one
        # ToDo: Optimize for getDirectionalDerivatives with seed vector Δx
        if fmu.jac_x != x || fmu.jac_t != t 

            if size(fmu.jac_dxy_x) != (length(rdx), length(rx))
                fmu.jac_dxy_x = fmi2GetJacobian(fmu, rdx, rx)
            else 
                fmi2GetJacobian!(fmu.jac_dxy_x, fmu, rdx, rx)
            end

            if size(fmu.jac_dxy_u) != (length(rdx), length(ru))
                fmu.jac_dxy_u = fmi2GetJacobian(fmu, rdx, ru)
            else 
                fmi2GetJacobian!(fmu.jac_dxy_u, fmu, rdx, ru)
            end

            fmu.jac_x = x
            fmu.jac_t = t
        end

        n_dx_x = fmu.jac_dxy_x * Δx
        n_dx_u = NoTangent()

        if setter
            n_dx_u = fmu.jac_dxy_u * ΔsetValues
        end

        f̄mu = ZeroTangent()
        x̄ = n_dx_x 
        t̄ = ZeroTangent()
        s̄etValueReferences = ZeroTangent()
        s̄etValues = n_dx_u
        ḡetValueReferences = ZeroTangent()
       
        return (f̄mu, x̄, t̄, s̄etValueReferences, s̄etValues, ḡetValueReferences)
    end
    return (y, fmi2DoStepME_pullforward(Δx, ΔsetValues)...)
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
    
    @assert fmu.modelDescription.isCoSimulation == fmi2True ["fmi2DoStepCS(...): As in the name, this function only supports CS-FMUs."]
    @assert length(setValueReferences) == length(setValues) ["fmi2DoStepCS(...): `setValueReferences` ($(length(setValueReferences))) and `setValues` ($(length(setValues))) need to be the same length!"]

    if length(setValueReferences) > 0
        fmi2SetReal(fmu, setValueReferences, setValues)
    end

    t = fmu.t
    fmi2DoStep(fmu, t, dt)
    fmu.t += dt

    y = zeros(Float64, 0)

    if length(getValueReferences) > 0
        y = fmi2GetReal(fmu, getValueReferences)
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
                              dt::Real = -1.0,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y = fmi2DoStepCS(fmu, dt, setValueReferences, setValues, getValueReferences)
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
            mat = fmi2GetJacobian(fmu, rdx, rx)
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

function ChainRulesCore.frule((Δself, Δfmu, Δdt, ΔsetValueReferences, ΔsetValues, ΔgetValueReferences), 
                              ::typeof(fmi2DoStepCS), 
                              fmu, #::FMU2,
                              dt,#::Real = -1.0,
                              setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                              setValues::Array{<:Real} = zeros(Real, 0),
                              getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    y = fmi2DoStepCS(fmu, dt, setValueReferences, setValues, getValueReferences)
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
            mat = fmi2GetJacobian(fmu, rdx, rx)
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