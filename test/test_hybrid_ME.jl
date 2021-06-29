#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using DifferentialEquations: Tsit5

modelFMUPath = joinpath(dirname(@__FILE__), "../model/SpringPendulum1D.fmu")
realFMUPath = joinpath(dirname(@__FILE__), "../model/SpringFrictionPendulum1D.fmu")

t_start = 0.0
t_step = 0.01
t_stop = 5.0

# generate training data
myFMU = fmiLoad(realFMUPath)
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)
x0 = fmi2GetContinuousStates(myFMU)
realSimData = fmi2SimulateCS(myFMU, t_step, t_start, t_stop, ["mass.s", "mass.v", "mass.f", "mass.a"], false)
fmiUnload(myFMU)

# load FMU for NeuralFMU
myFMU = fmiLoad(modelFMUPath)
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

# setup traing data
tData = t_start:t_step:t_stop
posData = fmi2SimulationResultGetValues(realSimData, "mass.s")
velData = fmi2SimulationResultGetValues(realSimData, "mass.v")

# loss function for training
function losssum()
    solution = problem(t_start, x0)

    posNet = collect(data[2] for data in solution.u)
    velNet = collect(data[3] for data in solution.u)

    Flux.Losses.mse(posNet, posData) + Flux.Losses.mse(velNet, velData)
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
numStates = fmiGetNumberOfStates(myFMU)
net = Chain(inputs -> fmi2DoStepME(myFMU, inputs),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))

problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(), tData)
@test problem != nothing

solutionBefore = problem(t_start, x0)
@test length(solutionBefore.t) == length(tData)
@test solutionBefore.t[1] == t_start
@test solutionBefore.t[end] == t_stop

# train it ...
p_net = Flux.params(problem)

optim = ADAM()
for i in 1:3
    @info "Epoch: $i / 3"
    Flux.train!(losssum, p_net, Iterators.repeated((), 500), optim; cb=callb)
end
@test losssum() < 0.1

# check results
solutionAfter = problem(t_start, x0)
@test length(solutionAfter.t) == length(tData)
@test solutionAfter.t[1] == t_start
@test solutionAfter.t[end] == t_stop

fmiUnload(myFMU)
