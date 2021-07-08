#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using DifferentialEquations: Tsit5

FMUPath = joinpath(dirname(@__FILE__), "..", "model", "SpringPendulumExtForce1D.fmu")

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate traing data
myFMU = fmiLoad(FMUPath)
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiSetReal(myFMU, "mass_s0", 1.3)   # increase amplitude, invert phase
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)
realSimData = fmiSimulateCS(myFMU, t_start, t_stop; recordValues=["mass.a"], setup=false, saveat=tData)

# reset FMU for use as NeuralFMU
fmiReset(myFMU)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

# setup traing data
extF = zeros(length(tData)) # no external force
accData = fmi2SimulationResultGetValues(realSimData, "mass.a")

# loss function for training
function losssum()
    solution = problem(t_step; inputs=extF)

    accNet = collect(data[2] for data in solution)

    Flux.Losses.mse(accNet, accData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb()
    global iterCB += 1
    global lastLoss

    if iterCB == 1
        lastLoss = losssum()
    end

    if iterCB % 50 == 0
        loss = losssum()
        @info "Loss: $loss"
        @test loss < lastLoss   
        lastLoss = loss
    end
end

# NeuralFMU setup
numInputs = length(myFMU.modelDescription.inputValueReferences)
numOutputs = length(myFMU.modelDescription.outputValueReferences)

net = Chain(inputs -> fmiInputDoStepCSOutput(myFMU, t_step, inputs),
            Dense(numOutputs, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numOutputs))

problem = CS_NeuralFMU(myFMU, net, (t_start, t_stop); saveat=tData)
@test problem != nothing

solutionBefore = problem(t_step; inputs=extF)
ts = collect(data[1] for data in solutionBefore)
@test length(ts) == length(tData)
@test abs(ts[1] - (t_start+t_step)) < 1e-10
@test abs(ts[end] - (t_stop+t_step)) < 1e-10

# train it ...
p_net = Flux.params(problem)

optim = ADAM()
Flux.train!(losssum, p_net, Iterators.repeated((), 300), optim; cb=callb)

# check results
solutionAfter = problem(t_step; inputs=extF)
ts = collect(data[1] for data in solutionAfter)
@test length(ts) == length(tData)
@test abs(ts[1] - (t_start+t_step)) < 1e-10
@test abs(ts[end] - (t_stop+t_step)) < 1e-10

fmiUnload(myFMU)
