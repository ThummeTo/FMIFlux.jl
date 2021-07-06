#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using DifferentialEquations: Tsit5

modelFMUPath = joinpath(dirname(@__FILE__), "..", "model", "SpringPendulum1D.fmu")
realFMUPath = joinpath(dirname(@__FILE__), "..", "model", "SpringFrictionPendulum1D.fmu")

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
myFMU = fmiLoad(realFMUPath)
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)
x0 = fmiGetContinuousStates(myFMU)
realSimData = fmiSimulateCS(myFMU, t_start, t_stop; recordValues=["mass.s", "mass.v"], setup=false, saveat=tData)
fmiUnload(myFMU)

# load FMU for NeuralFMU
myFMU = fmiLoad(modelFMUPath)
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

# setup traing data
posData = fmi2SimulationResultGetValues(realSimData, "mass.s")
#velData = fmi2SimulationResultGetValues(realSimData, "mass.v")

# loss function for training
function losssum()
    global problem, x0
    solution = problem(x0)

    posNet = collect(data[2] for data in solution.u)
    #velNet = collect(data[3] for data in solution.u)

    Flux.Losses.mse(posNet, posData) #+ Flux.Losses.mse(velNet, velData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb()
    global iterCB += 1
    global lastLoss

    if iterCB % 10 == 0
        loss = losssum()
        @info "Loss: $loss"
        @test loss < lastLoss     
        lastLoss = loss
    end
end

vr = fmi2String2ValueReference(myFMU, "mass.m")

numStates = fmiGetNumberOfStates(myFMU)

# some NeuralFMU setups
nets = [] 

# 1 # default ME-NeuralFMU (learn dynamics)
net = Chain(states ->  fmiDoStepME(myFMU, states), 
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 2 # default ME-NeuralFMU (learn states)
net = Chain(Dense(numStates, 16, leakyrelu),
            Dense(16, 16, leakyrelu),
            Dense(16, numStates),
            states ->  fmiDoStepME(myFMU, states))
push!(nets, net)

# 3 # default ME-NeuralFMU (learn dynamics and states)
net = Chain(Dense(numStates, 16, leakyrelu),
            Dense(16, 16, leakyrelu),
            Dense(16, numStates),
            states ->  fmiDoStepME(myFMU, states),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 4 # NeuralFMU with hard setting time to 0.0
net = Chain(states ->  fmiDoStepME(myFMU, states, 0.0), 
            Dense(numStates, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 5 # NeuralFMU with additional getter 
getVRs = [fmi2String2ValueReference(myFMU, "mass.m")]
numGetVRs = length(getVRs)
net = Chain(states ->  fmiDoStepME(myFMU, states, -1.0, [], [], getVRs), 
            Dense(numStates+numGetVRs, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 6 # NeuralFMU with additional setter 
setVRs = [fmi2String2ValueReference(myFMU, "mass.m")]
numSetVRs = length(setVRs)
net = Chain(states ->  fmiDoStepME(myFMU, states, -1.0, setVRs, [1.1]), 
            Dense(numStates, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 7 # NeuralFMU with additional setter and getter
net = Chain(states ->  fmiDoStepME(myFMU, states, -1.0, setVRs, [1.1], getVRs), 
            Dense(numStates+numGetVRs, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

optim = ADAM()
for i in 1:length(nets)
    @testset "Net setup #$i" begin
        global nets, problem, lastLoss, iterCB
        net = nets[i]
        problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(); saveat=tData)
        @test problem != nothing

        solutionBefore = problem(x0)
        @test length(solutionBefore.t) == length(tData)
        @test solutionBefore.t[1] == t_start
        @test solutionBefore.t[end] == t_stop

        # train it ...
        p_net = Flux.params(problem)

        iterCB = 0
        lastLoss = losssum()

        Flux.train!(losssum, p_net, Iterators.repeated((), 100), optim; cb=callb)

        # check results
        solutionAfter = problem(x0)
        @test length(solutionAfter.t) == length(tData)
        @test solutionAfter.t[1] == t_start
        @test solutionAfter.t[end] == t_stop
    end
end

fmiUnload(myFMU)
