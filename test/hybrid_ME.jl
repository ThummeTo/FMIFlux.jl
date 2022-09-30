#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using DifferentialEquations: Tsit5

import Random 
Random.seed!(1234);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
realFMU = fmiLoad("SpringFrictionPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"])
fmiInstantiate!(realFMU; loggingOn=false)
fmiSetupExperiment(realFMU, t_start, t_stop)
fmiEnterInitializationMode(realFMU)
fmiExitInitializationMode(realFMU)
x0 = fmiGetContinuousStates(realFMU)
realSimData = fmiSimulateCS(realFMU, t_start, t_stop; recordValues=["mass.s", "mass.v"], setup=false, reset=false, instantiate=false, saveat=tData)

# load FMU for NeuralFMU
myFMU = fmiLoad("SpringPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"])
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

# setup traing data
posData = fmi2GetSolutionValue(realSimData, "mass.s")

# loss function for training
function losssum(p)
    global problem, x0, posData
    solution = problem(x0; p=p)

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    Flux.Losses.mse(posNet, posData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb(p)
    global iterCB += 1
    global lastLoss

    if iterCB % 30 == 0
        loss = losssum(p[1])
        @info "Loss: $loss"
        @test loss < lastLoss  
        lastLoss = loss
    end
end

numStates = fmiGetNumberOfStates(myFMU)

# some NeuralFMU setups
nets = [] 

1 # default ME-NeuralFMU (learn dynamics and states, almost-neutral setup, parameter count << 100)
net = Chain(Dense(numStates, numStates, tanh; init=Flux.identity_init),
            states ->  fmiEvaluateME(myFMU, states), 
            Dense(numStates, numStates, identity; init=Flux.identity_init))
push!(nets, net)

# 2 # default ME-NeuralFMU (learn dynamics)
net = Chain(states ->  fmiEvaluateME(myFMU, states), 
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 3 # default ME-NeuralFMU (learn states)
net = Chain(Dense(numStates, 16, identity),
            Dense(16, 16, identity),
            Dense(16, numStates),
            states -> fmiEvaluateME(myFMU, states))
push!(nets, net)

# 4 # default ME-NeuralFMU (learn dynamics and states)
net = Chain(Dense(numStates, 16, leakyrelu),
            Dense(16, 16, leakyrelu),
            Dense(16, numStates),
            states -> fmiEvaluateME(myFMU, states),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 5 # NeuralFMU with hard setting time to 0.0
net = Chain(states ->  fmiEvaluateME(myFMU, states), # not supported by this FMU:   states ->  fmiEvaluateME(myFMU, states, 0.0), 
            Dense(numStates, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 6 # NeuralFMU with additional getter 
getVRs = [fmi2StringToValueReference(myFMU, "mass.m")]
numGetVRs = length(getVRs)
net = Chain(states ->  fmiEvaluateME(myFMU, states, myFMU.components[end].t, fmi2ValueReference[], Real[], getVRs), 
            Dense(numStates+numGetVRs, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 7 # NeuralFMU with additional setter 
setVRs = [fmi2StringToValueReference(myFMU, "mass.m")]
numSetVRs = length(setVRs)
net = Chain(states ->  fmiEvaluateME(myFMU, states, myFMU.components[end].t, setVRs, [1.1]), 
            Dense(numStates, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 8 # NeuralFMU with additional setter and getter
net = Chain(states ->  fmiEvaluateME(myFMU, states, myFMU.components[end].t, setVRs, [1.1], getVRs), 
            Dense(numStates+numGetVRs, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

optim = Adam(1e-4)
for i in 1:length(nets)
    @testset "Net setup #$i" begin
        global nets, problem, lastLoss, iterCB
        net = nets[i]
        problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(); saveat=tData)
        @test problem != nothing

        solutionBefore = problem(x0)
        if solutionBefore.success
            @test length(solutionBefore.states.t) == length(tData)
            @test solutionBefore.states.t[1] == t_start
            @test solutionBefore.states.t[end] == t_stop
        end

        # train it ...
        p_net = Flux.params(problem)

        iterCB = 0
        lastLoss = losssum(p_net[1])
        @info "Start-Loss for net #$i: $lastLoss"
        FMIFlux.train!(losssum, p_net, Iterators.repeated((), 60), optim; cb=()->callb(p_net))

        # check results
        solutionAfter = problem(x0)
        if solutionAfter.success
            @test length(solutionAfter.states.t) == length(tData)
            @test solutionAfter.states.t[1] == t_start
            @test solutionAfter.states.t[end] == t_stop
        end
    end
end

fmiUnload(realFMU)
fmiUnload(myFMU)
