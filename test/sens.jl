#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using DifferentialEquations: Tsit5

import ForwardDiff
import Zygote

using FMIFlux.FMIImport: fmi2SampleDirectionalDerivative, fmi2GetJacobian, fmi2SetContinuousStates

FMUPaths = [joinpath(dirname(@__FILE__), "..", "model", "SpringFrictionPendulum1D.fmu"),
            joinpath(dirname(@__FILE__), "..", "model", "BouncingBall1D.fmu")]

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

for FMUPath in FMUPaths
    myFMU = fmiLoad(FMUPath)
    comp = fmiInstantiate!(myFMU; loggingOn=false)
    fmiSetupExperiment(myFMU, t_start, t_stop)
    fmiEnterInitializationMode(myFMU)
    fmiExitInitializationMode(myFMU)

    x0 = fmiGetContinuousStates(myFMU)
    numStates = length(x0)

    # Jacobians for x0
    FD_jac = ForwardDiff.jacobian(x -> fmiEvaluateME(myFMU, x, 0.0), x0)
    ZG_jac = Zygote.jacobian(fmiEvaluateME, myFMU, x0, 0.0)[2]
    fmiSetContinuousStates(myFMU, x0)
    samp_jac = fmi2SampleDirectionalDerivative(myFMU, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)
    auto_jac = fmi2GetJacobian(myFMU, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)

    @test (abs.(auto_jac -   FD_jac) .< ones(numStates, numStates).*1e-6) == ones(Bool, numStates, numStates)
    @test (abs.(auto_jac -   ZG_jac) .< ones(numStates, numStates).*1e-6) == ones(Bool, numStates, numStates)
    @test (abs.(auto_jac - samp_jac) .< ones(numStates, numStates).*1e-6) == ones(Bool, numStates, numStates)

    # Jacobians for random x0 
    x0 = x0 + rand(numStates)
    FD_jac = ForwardDiff.jacobian(x -> fmiEvaluateME(myFMU, x, 0.0), x0)
    ZG_jac = Zygote.jacobian(fmiEvaluateME, myFMU, x0, 0.0)[2]
    fmi2SetContinuousStates(myFMU, x0)
    samp_jac = fmi2SampleDirectionalDerivative(myFMU, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)
    auto_jac = fmi2GetJacobian(myFMU, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)

    @test (abs.(auto_jac -   FD_jac) .< ones(numStates, numStates).*1e-6) == ones(Bool, numStates, numStates)
    @test (abs.(auto_jac -   ZG_jac) .< ones(numStates, numStates).*1e-6) == ones(Bool, numStates, numStates)
    @test (abs.(auto_jac - samp_jac) .< ones(numStates, numStates).*1e-6) == ones(Bool, numStates, numStates)

    fmiUnload(myFMU)
end
