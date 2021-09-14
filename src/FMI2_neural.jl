#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# helper, builds jacobian (d dx / d x) for FMUs
function _build_jac_dx_x(fmu::FMU2, rdx, rx)
    mat = zeros(length(rdx), length(rx))

    for i in 1:length(rdx)
        for j in 1:length(rx)
            mat[i,j] = fmi2GetDirectionalDerivative(fmu, rdx[i], rx[j])
        end
    end

    mat
end

"""
Performs the equivalent of fmiDoStep for ME-FMUs (note, that fmiDoStep is for CS-FMUs only).
Currently no event handling supported (but often not necessary).
Note, that event-handling during ME simulation in FMI.jl itself is supported.

Optional, FMU-values can be set via keyword arguments `setValueReferences` and `setValues`.
Optional, FMU-values can be retrieved by keyword argument `getValueReferences`.

Function returns an array with state derivatives and optionally the FMU-values for `getValueReferences`.
"""
function fmi2DoStepME(fmu::FMU2,
    x::Array,
    t::Real = -1.0,
    setValueReferences=[],
    setValues=[],
    getValueReferences=[])
    if t < 0.0
        t = NeuralFMUGetCachedTime(fmu)
    end
    _fmi2DoStepME(fmu, x, t, setValueReferences, setValues, getValueReferences)
end
# function fmi2DoStepME(fmu::FMU2,
#                       x::Array,
#                       t::Real = -1.0;
#                       setValueReferences=[],
#                       setValues=[],
#                       getValueReferences=[])
#     if t < 0.0
#         t = NeuralFMUGetCachedTime(fmu)
#     end
#     _fmi2DoStepME(fmu, x, t, setValueReferences, setValues, getValueReferences)
# end

# helper because keyword arguments are not supported by Zygote
function _fmi2DoStepME(fmu::FMU2,
                       x,
                       t::Real,
                       setValueReferences = [],
                       setValues = [],
                       getValueReferences = [])

    @assert fmi2IsModelExchange(fmu) ["fmi2DoStepME(...): As in the name, this function only supports ME-FMUs."]

    setter = (length(setValueReferences) > 0)
    getter = (length(getValueReferences) > 0)

    if setter
        @assert setValues != nothing ["fmi2DoStepME(...): `setValues` is nothing but `setValueReferences` is not!"]
        @assert length(setValueReferences) == length(setValues) ["fmi2DoStepME(...): `setValueReferences` and `setValues` need to be the same length!"]
    end

    fmi2SetTime(fmu, t)

    if setter
        fmi2SetReal(fmu, setValueReferences, setValues)
    end

    fmi2SetContinuousStates(fmu, x)

    fmi2CompletedIntegratorStep(fmu, fmi2True)

    y = []
    if getter
        y = fmi2GetReal(fmu, getValueReferences)
    end

    dx = fmi2GetDerivatives(fmu)

    [dx..., y...] # vcat(dx, y)
end

# The gradient for the function fmi2SetDoStepMEGet.
function _fmi2DoStepME_Gradient(c̄,
                                fmu::FMU2,
                                x,
                                t::Real,
                                setValueReferences = [],
                                setValues = [],
                                getValueReferences = [])

    setter = (length(setValueReferences) > 0)
    getter = (length(getValueReferences) > 0)

    if setter
        @assert setValues != nothing ["_fmi2DoStepME_Gradient(...): `setValues` is nothing but `setValueReferences` is not!"]
        @assert length(setValueReferences) == length(setValues) ["_fmi2DoStepME_Gradient(...): `setValueReferences` and `setValues` need to be the same length!"]
    end

    fmi2SetTime(fmu, t)
    fmi2SetContinuousStates(fmu, x)

    rdx = vcat(fmu.modelDescription.derivativeValueReferences, getValueReferences) #rdx = [fmu.modelDescription.derivativeValueReferences..., getValueReferences...]
    rx = fmu.modelDescription.stateValueReferences
    ru = setValueReferences

    n_dx_x = zeros(Float64, 0)
    if length(x) > 0
        mat_dx_x = _build_jac_dx_x(fmu, rdx, rx)
        n_dx_x = mat_dx_x' * c̄
    end

    n_dx_u = zeros(Float64, 0)
    if setter
        mat_dx_u = _build_jac_dx_x(fmu, rdx, ru)
        n_dx_u = mat_dx_u' * c̄
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
"""
function fmi2DoStepCS(fmu::FMU2, dt; setValueReferences::Array = [], setValues::Array = [], getValueReferences::Array = [])
    _fmi2DoStepCS(fmu, dt, setValueReferences, setValues, getValueReferences)
end

function _fmi2DoStepCS(fmu::FMU2, dt, setValueReferences::Array = [], setValues::Array = [], getValueReferences::Array = [])
    
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
function _fmi2DoStepCS_Gradient(c̄, fmu::FMU2, dt, setValueReferences = [], setValues = [], getValueReferences = [])
    
    rdx = getValueReferences
    rx = setValueReferences

    n = zeros(Float64, 0)

    if length(getValueReferences) > 0
        mat =  _build_jac_dx_x(fmu, rdx, rx)
        n = mat' * c̄
    end

    nvr = zeros(Float64, length(setValueReferences))
    gvr = zeros(Float64, length(getValueReferences))

    tuple(nothing, 0.0, nvr, n, gvr)
end

# The adjoint connection between CS-function and function gradient.
@adjoint _fmi2DoStepCS(fmu, dt, setValueReferences, setValues, getValueReferences) = _fmi2DoStepCS(fmu, dt, setValueReferences, setValues, getValueReferences), c̄ -> _fmi2DoStepCS_Gradient(c̄, fmu, dt, setValueReferences, setValues, getValueReferences)

"""
Sets all FMU inputs to u, performs a ´´´fmi2DoStep´´´ and returns all FMU outputs.
"""
function fmi2InputDoStepCSOutput(fmu::FMU2, dt, u)
    _fmi2DoStepCS(fmu, dt,
                 fmu.modelDescription.inputValueReferences,
                 u,
                 fmu.modelDescription.outputValueReferences)
end

function fmi2InputDoStepCSOutput_Gradient(c̄, fmu::FMU2, dt, u)
    _fmi2DoStepCS_Gradient(c̄, fmu, dt,
                           fmu.modelDescription.inputValueReferences,
                           u,
                           fmu.modelDescription.outputValueReferences)
end

# The adjoint connection between CS-function and function gradient.
@adjoint fmi2InputDoStepCSOutput(fmu, dt, u) = fmi2InputDoStepCSOutput(fmu, dt, u), c̄ -> fmi2InputDoStepCSOutput_Gradient(c̄, fmu, dt, u)