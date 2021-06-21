#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using DifferentialEquations: Tsit5

FMUPath = joinpath(dirname(@__FILE__), "../model/SpringPendulumExtForce1D.fmu")

t_start = 0.0
t_step = 0.01
t_stop = 5.0

# generate traing data
myFMU = fmiLoad(FMUPath)
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiSetReal(myFMU, "mass_s0", 1.3)   # increase amplitude, invert phase
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)
realSimData = fmi2SimulateCS(myFMU, t_step, t_start, t_stop, ["mass.a"], false)

# reset FMU for use as NeuralFMU
fmiReset(myFMU)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

# setup traing data
tData = t_start:t_step:t_stop
extF = zeros(length(tData)) # no external force
accData = fmi2SimulationResultGetValues(realSimData, "mass.a")

# loss function for training
function losssum()
    solution = problem(t_start, t_step, t_stop, extF)

    accNet = collect(data[2] for data in solution)

    Flux.Losses.mse(accNet, accData)
end

# callback function for training
global iterCB = 0
function callb()
    global iterCB += 1

    if iterCB % 10 == 1
        avg_ls = losssum()
        @info "Loss: $(round(avg_ls, digits=5))"
    end
end

# NeuralFMU setup
numInputs = length(myFMU.modelDescription.inputValueReferences)
numOutputs = length(myFMU.modelDescription.outputValueReferences)

net = Chain(inputs -> fmi2InputDoStepCSOutput(myFMU, t_step, inputs),
            Dense(numOutputs, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numOutputs))

problem = CS_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(), tData, true, true)
@test problem != nothing

solutionBefore = problem(t_start, t_step, t_stop, extF)
ts = collect(data[1] for data in solutionBefore)
@test length(ts) == length(tData)
@test abs(ts[1] - (t_start + t_step)) < 1e-10
@test abs(ts[end] - (t_stop + t_step)) < 1e-10

# train it ...
p_net = Flux.params(problem)

optim = ADAM()
Flux.train!(losssum, p_net, Iterators.repeated((), 300), optim; cb=callb)
@test losssum() < 0.01

# check results
solutionAfter = problem(t_start, t_step, t_stop, extF)
ts = collect(data[1] for data in solutionAfter)
@test length(ts) == length(tData)
@test abs(ts[1] - (t_start + t_step)) < 1e-10
@test abs(ts[end] - (t_stop + t_step)) < 1e-10

fmiUnload(myFMU)
