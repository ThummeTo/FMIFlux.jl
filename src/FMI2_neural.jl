#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using DiffEqFlux: ForwardDiff

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

    depMtx = fmi2GetDependencies(fmu)
    rdx_inds = collect(fmu.modelDescription.valueReferenceIndicies[vr] for vr in rdx)
    rx_inds  = collect(fmu.modelDescription.valueReferenceIndicies[vr] for vr in rx)
    
    for i in 1:length(rx)

        sensitive_rdx_inds = Int64[]
        sensitive_rdx = fmi2ValueReference[]

        for j in 1:length(rdx)
            if depMtx[rdx_inds[j], rx_inds[i]] != fmi2DependencyIndependent
                push!(sensitive_rdx_inds, j)
                push!(sensitive_rdx, rdx[j])
            end
        end

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
Performs the equivalent of fmiDoStep for ME-FMUs (note, that fmiDoStep is for CS-FMUs only).
Currently no event handling supported (but often not necessary, if the training loop is robust against small/short errors in the loss gradient).
Note, that event-handling during ME simulation in FMI.jl itself is supported.

Optional, additional FMU-values can be set via keyword arguments `setValueReferences` and `setValues`.
Optional, additional FMU-values can be retrieved by keyword argument `getValueReferences`.

Function takes the current system state array ("x") and returns an array with state derivatives ("x dot") and optionally the FMU-values for `getValueReferences`.
Setting the FMU time via argument `t` is optional, if not set, the current time of the ODE solver around the NeuralFMU is used.
"""
function fmi2DoStepME(fmu::FMU2,
                      x::Array{<:Real},
                      t::Real = -1.0,
                      setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                      setValues::Array{<:Real} = zeros(Real, 0),
                      getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    if t < 0.0
        t = NeuralFMUGetCachedTime(fmu)
    end
    _fmi2DoStepME(fmu, x, t, setValueReferences, setValues, getValueReferences)
end

# Helper because keyword arguments are (currently) not supported by Zygote.
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

    fmi2SetTime(fmu, t)

    if setter
        fmi2SetReal(fmu, setValueReferences, setValues)
    end

    # small parameter counts lead to the use of ForwardDiff, state vector needs to be translated back into Float64 
    if all(isa.(x, ForwardDiff.Dual))
        xf = collect(ForwardDiff.value(e) for e in x)
        fmi2SetContinuousStates(fmu, xf)
    else
        fmi2SetContinuousStates(fmu, x)
    end

    fmi2CompletedIntegratorStep(fmu, fmi2True)

    y = []
    if getter
        y = fmi2GetReal(fmu, getValueReferences)
    end

    dx = fmi2GetDerivatives(fmu)

    [dx..., y...] # vcat(dx, y)
end

# The gradient for the function fmi2DoStepME.
function _fmi2DoStepME_Gradient(c̄,
                                fmu::FMU2,
                                x::Array{<:Real},
                                t::Real,
                                setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                                setValues::Array{<:Real} = zeros(Real, 0),
                                getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))

    setter = (length(setValueReferences) > 0)
    getter = (length(getValueReferences) > 0)

    if setter
        @assert length(setValueReferences) == length(setValues) ["_fmi2DoStepME_Gradient(...): `setValueReferences` and `setValues` need to be the same length!"]
    end

    fmi2SetTime(fmu, t)
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

    n_dx_x = fmu.jac_dxy_x' * c̄
    n_dx_u = nothing

    if setter
        n_dx_u = fmu.jac_dxy_u' * c̄
    end

    svr = 0.0 # zeros(Float64, length(setValueReferences), length(rdx)) * c̄
    gvr = 0.0 #zeros(Float64, length(getValueReferences), length(rdx)) * c̄

    return tuple(0.0, n_dx_x, 0.0, svr, n_dx_u, gvr)
end

# The adjoint connection between ME-function and function gradient.
@adjoint _fmi2DoStepME(fmu, x, t, setValueReferences, setValues, getValueReferences) = _fmi2DoStepME(fmu, x, t, setValueReferences, setValues, getValueReferences), c̄ -> _fmi2DoStepME_Gradient(c̄, fmu, x, t, setValueReferences, setValues, getValueReferences)

"""
Performs a fmiDoStep for CS-FMUs (note, that fmiDoStep is for CS-FMUs only).

Optional, FMU-values can be set via keyword arguments `setValueReferences` and `setValues`.
Optional, FMU-values can be retrieved by keyword argument `getValueReferences`.

Function returns the FMU-values for the optional `getValueReferences`.
The CS-FMU performs one macro step with step size `dt`. Dependent on the integrated numerical solver, the FMU may perform multiple (internal) micro steps if needed to meet solver requirements (stability/accuracy). These micro steps are hidden by FMI.
"""
function fmi2DoStepCS(fmu::FMU2, 
                      dt::Real; 
                      setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                      setValues::Array{<:Real} = zeros(Real, 0),
                      getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    _fmi2DoStepCS(fmu, dt, setValueReferences, setValues, getValueReferences)
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

# The gradient for the function fmi2InputDoStepCSOutput.
function _fmi2DoStepCS_Gradient(c̄, 
                                fmu::FMU2, 
                                dt::Real, 
                                setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0),
                                setValues::Array{<:Real} = zeros(Real, 0),
                                getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    
    rdx = getValueReferences
    rx = setValueReferences

    n = zeros(Float64, 0)

    if length(getValueReferences) > 0
        mat = fmi2GetJacobian(fmu, rdx, rx)
        n = mat' * c̄
    end

    nvr = zeros(Float64, length(setValueReferences))
    gvr = zeros(Float64, length(getValueReferences))

    tuple(nothing, 0.0, nvr, n, gvr)
end

# The adjoint connection between CS-function and function gradient.
@adjoint _fmi2DoStepCS(fmu, dt, setValueReferences, setValues, getValueReferences) = _fmi2DoStepCS(fmu, dt, setValueReferences, setValues, getValueReferences), c̄ -> _fmi2DoStepCS_Gradient(c̄, fmu, dt, setValueReferences, setValues, getValueReferences)

"""
Sets all FMU inputs to `u`, performs a ´´´fmi2DoStep´´´ and returns all FMU outputs.
"""
function fmi2InputDoStepCSOutput(fmu::FMU2, 
                                 dt::Real, 
                                 u::Array{<:Real})
                                 
    @assert fmi2IsCoSimulation(fmu) ["fmi2InputDoStepCSOutput(...): As in the name, this function only supports CS-FMUs."]

    _fmi2DoStepCS(fmu, dt,
                 fmu.modelDescription.inputValueReferences,
                 u,
                 fmu.modelDescription.outputValueReferences)
end

# Helper because keyword arguments are (currently) not supported by Zygote.
function fmi2InputDoStepCSOutput_Gradient(c̄, 
                                          fmu::FMU2, 
                                          dt::Real, 
                                          u::Array{<:Real})
    _fmi2DoStepCS_Gradient(c̄, fmu, dt,
                           fmu.modelDescription.inputValueReferences,
                           u,
                           fmu.modelDescription.outputValueReferences)
end

# The adjoint connection between CS-function and function gradient.
@adjoint fmi2InputDoStepCSOutput(fmu, dt, u) = fmi2InputDoStepCSOutput(fmu, dt, u), c̄ -> fmi2InputDoStepCSOutput_Gradient(c̄, fmu, dt, u)