#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Zygote: @adjoint
using Zygote
using Flux, DiffEqFlux
using OrdinaryDiffEq

include("FMI2_neural.jl")

mutable struct ME_NeuralFMU
    neuralODE::NeuralODE
    solution::ODESolution
    fmu

    ME_NeuralFMU() = new()
end

mutable struct CS_NeuralFMU
    model
    simulationResult::fmi2SimulationResult
    fmu

    CS_NeuralFMU() = new()
end

NeuralFMU = ME_NeuralFMU
NeuralFMUs = Union{ME_NeuralFMU, CS_NeuralFMU}

# time caching (to set correct time in ME-NeuralFMUs)
function NeuralFMUCacheTime(fmu, t)
    fmu.next_t = t
end
function NeuralFMUCacheTime_Gradient(c̄, fmu, t)
    tuple(0.0, 0.0)
end
@adjoint NeuralFMUCacheTime(fmu, t) = NeuralFMUCacheTime(fmu, t), c̄ -> NeuralFMUCacheTime(c̄, fmu, t)

# helper to add an additional time state later used to setup time inside the FMU
function NeuralFMUInputLayer(fmu, inputs)
    t = inputs[1]
    x = inputs[2:end]
    NeuralFMUCacheTime(fmu, t)
    x
end

# helper to add an additional time state later used to setup time inside the FMU
function NeuralFMUOutputLayerME(inputs)
    dt = 1.0
    dx = inputs
    vcat([dt], dx)
end

function NeuralFMUOutputLayerCS(fmu, inputs)
    out = inputs
    t = Zygote.@ignore fmu.t
    vcat([t], out)
end

"""
Constructs a ME-NeuralFMU where the FMU is at an unknown location inside of the NN.
"""
function ME_NeuralFMU(fmu, model, tspan, alg=nothing, saveat=[], addTopLayer = true, addBottomLayer = true)
    nfmu = ME_NeuralFMU()
    nfmu.fmu = fmu

    ext_model = nothing

    if addTopLayer && addBottomLayer
        ext_model = Chain(inputs -> NeuralFMUInputLayer(nfmu.fmu, inputs),
                          model.layers...,
                          inputs -> NeuralFMUOutputLayerME(inputs))
    elseif addTopLayer
        ext_model = Chain(inputs -> NeuralFMUInputLayer(nfmu.fmu, inputs),
                          model.layers...)
    elseif addBottomLayer
        ext_model = Chain(model.layers...,
                          inputs -> NeuralFMUOutputLayerME(inputs))
    else
        ext_model = model.layers
    end

    nfmu.neuralODE = NeuralODE(ext_model, tspan, alg, saveat=saveat) #, p=Flux.params(ext_model))

    nfmu
end

"""
Constructs a CS-NeuralFMU where the FMU is at an unknown location inside of the NN.
"""
function CS_NeuralFMU(fmu, model, tspan, alg=nothing, saveat=[], addTopLayer = true, addBottomLayer = true, recordValues = [])
    nfmu = CS_NeuralFMU()
    nfmu.fmu = fmu

    if addTopLayer && addBottomLayer
        nfmu.model = Chain(inputs -> NeuralFMUInputLayer(nfmu.fmu, inputs),
                          model.layers...,
                          inputs -> NeuralFMUOutputLayerCS(nfmu.fmu, inputs))
    elseif addTopLayer
        nfmu.model = Chain(inputs -> NeuralFMUInputLayer(nfmu.fmu, inputs),
                          model.layers...)
    elseif addBottomLayer
        nfmu.model = Chain(model.layers...,
                          inputs -> NeuralFMUOutputLayerCS(nfmu.fmu, inputs))
    else
        nfmu.model = model.layers
    end

    # allocate simulation result
    nfmu.simulationResult = fmi2SimulationResult()
    nfmu.simulationResult.valueReferences = nfmu.fmu.modelDescription.outputValueReferences # vcat(recordValues, nfmu.fmu.modelDescription.inputValueReferences, nfmu.fmu.modelDescription.outputValueReferences)
    nfmu.simulationResult.dataPoints = []
    nfmu.simulationResult.fmu = nfmu.fmu

    for t in saveat
        values = zeros(length(nfmu.simulationResult.valueReferences))
        push!(nfmu.simulationResult.dataPoints, [t, values...])
    end

    nfmu
end

function (nfmu::ME_NeuralFMU)(t0, x0, reset::Bool=false)
    nfmu([t0, x0...], reset) # vcat([t0], x0)
end

function (nfmu::ME_NeuralFMU)(val0, reset::Bool=false)

    t0 = val0[1]

    if reset
        fmiReset(nfmu.fmu)
        fmiSetupExperiment(nfmu.fmu, t0)
        fmiEnterInitializationMode(nfmu.fmu)
        fmiExitInitializationMode(nfmu.fmu)
    end

    nfmu.solution = nfmu.neuralODE(val0)

    nfmu.solution
end

function (nfmu::CS_NeuralFMU)(t_start, t_step, t_stop, inputs, reset::Bool=true)

    numIter = round((t_stop-t_start)/t_step)

    if reset
        while fmiReset(nfmu.fmu) != 0
        end
        while fmiSetupExperiment(nfmu.fmu, t_start) != 0
        end
        while fmiEnterInitializationMode(nfmu.fmu) != 0
        end
        while fmiExitInitializationMode(nfmu.fmu) != 0
        end
    end

    t = t_start

    ts = t_start:t_step:t_stop
    modelInput = collect.(eachrow(hcat(ts, inputs)))
    valueStack = nfmu.model.(modelInput)

    valueStack
end

# adapting the Flux functions
function Flux.params(neuralfmu::ME_NeuralFMU)
    Flux.params(neuralfmu.neuralODE)
end

function Flux.params(neuralfmu::CS_NeuralFMU)
    Flux.params(neuralfmu.model)
end

# FMI version independent dosteps
function fmiDoStepME(fmu::FMU2, t, x)
    fmi2DoStepME(fmu, t, x)
end
function fmiDoStepME(fmu::FMU2, x)
    fmi2DoStepME(fmu, x)
end

# FMI version independent dostep-gradients
function fmiDoStepME_Gradient(c̄, fmu::FMU2, t, x)
    fmi2DoStepME_Gradient(c̄, fmu, t, x)
end
function fmiDoStepME_Gradient(c̄, fmu::FMU2, x)
    fmi2DoStepME_Gradient(c̄, fmu, x)
end

@adjoint fmiDoStepME(fmu, t, x) = fmiDoStepME(fmu, t, x), c̄ -> fmiDoStepME_Gradient(c̄, fmu, t, x)
@adjoint fmiDoStepME(fmu, x) = fmiDoStepME(fmu, x), c̄ -> fmiDoStepME_Gradient(c̄, fmu, x)


# define neutral gradients (=feed-trough) for ccall-functions
function neutralGradient(c̄)
    c̄
end

@adjoint fmiSetupExperiment(fmu, startTime, stopTime) = fmiSetupExperiment(fmu, startTime, stopTime), c̄ -> neutralGradient(c̄)
@adjoint fmiEnterInitializationMode(fmu) = fmiEnterInitializationMode(fmu), c̄ -> neutralGradient(c̄)
@adjoint fmiExitInitializationMode(fmu) = fmiExitInitializationMode(fmu), c̄ -> neutralGradient(c̄)
@adjoint fmiReset(fmu) = fmiReset(fmu), c̄ -> neutralGradient(c̄)
