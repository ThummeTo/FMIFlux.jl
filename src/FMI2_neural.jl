#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

#using Zygote: @adjoint
#using Flux, DiffEqFlux

# helper, builds jacobian (d dx / d x) for ME-FMU
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
Currently no event handling supported.
"""
function fmi2DoStepME(fmu::FMU2, t, x)

    @assert fmu.modelDescription.isModelExchange == fmi2True ["fmi2DoStepME(...): As in the name, this function only supports ME-FMUs."]

    fmu.t = t
    fmi2SetTime(fmu, t)

    fmi2SetContinuousStates(fmu, x)

    fmi2CompletedIntegratorStep(fmu, fmi2True)

    dx = fmi2GetDerivatives(fmu)

    dx
end

function fmi2DoStepME(fmu::FMU2, x)
    t = fmu.next_t
    fmi2DoStepME(fmu, t, x)
end

# The gradient for the function fmi2DoStepME.
function fmi2DoStepME_Gradient(c̄, fmu::FMU2, t, x)

    fmu.t = t
    fmi2SetTime(fmu, t)

    fmi2SetContinuousStates(fmu, x)

    rdx = fmu.modelDescription.derivativeValueReferences
    rx = fmu.modelDescription.stateValueReferences

    mat =  _build_jac_dx_x(fmu, rdx, rx)

    n = mat' * c̄

    tuple(0.0, 0.0, n)
end

function fmi2DoStepME_Gradient(c̄, fmu::FMU2, x)
    t = fmu.t
    grad = fmi2DoStepME_Gradient(c̄, fmu::FMU2, t, x)
    grad[2:end] # remove one of the leading zeros
end

# The adjoint connection between ME-function and function gradient.
@adjoint fmi2DoStepME(fmu, t, x) = fmi2DoStepME(fmu, t, x), c̄ -> fmi2DoStepME_Gradient(c̄, fmu, t, x)
@adjoint fmi2DoStepME(fmu, x) = fmi2DoStepME(fmu, x), c̄ -> fmi2DoStepME_Gradient(c̄, fmu, x)

"""
Sets all FMU inputs to u, performs a ´´´fmi2DoStep´´´ and returns all FMU outputs.
"""
function fmi2InputDoStepCSOutput(fmu::FMU2, dt, u)

    @assert fmu.modelDescription.isCoSimulation == fmi2True ["fmi2InputDoStepCSOutput(...): As in the name, this function only supports CS-FMUs."]

    fmi2SetReal(fmu, fmu.modelDescription.inputValueReferences, u)

    t = fmu.t
    fmi2DoStep(fmu, t, dt)
    fmu.t += dt

    y = fmi2GetReal(fmu, fmu.modelDescription.outputValueReferences)

    y
end

# The gradient for the function fmi2InputDoStepCSOutput.
function fmi2InputDoStepCSOutput_Gradient(c̄, fmu::FMU2, dt, u)

    rdx = fmu.modelDescription.outputValueReferences
    rx = fmu.modelDescription.inputValueReferences

    mat =  _build_jac_dx_x(fmu, rdx, rx)

    n = mat' * c̄

    tuple(0.0, 0.0, n)
end

# The adjoint connection between CS-function and function gradient.
@adjoint fmi2InputDoStepCSOutput(fmu, dt, u) = fmi2InputDoStepCSOutput(fmu, dt, u), c̄ -> fmi2InputDoStepCSOutput_Gradient(c̄, fmu, dt, u)
