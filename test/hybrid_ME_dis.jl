#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using DifferentialEquations: Tsit5

import Random 
Random.seed!(5678);

t_start = 0.0
t_step = 0.1
t_stop = 3.0
tData = t_start:t_step:t_stop

# generate training data
realFMU = fmiLoad("BouncingBall1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"])
fmiInstantiate!(realFMU; loggingOn=false)
fmiSetupExperiment(realFMU, t_start, t_stop)
fmiEnterInitializationMode(realFMU)
fmiExitInitializationMode(realFMU)
x0 = fmiGetContinuousStates(realFMU)
realSimData = fmiSimulateCS(realFMU, t_start, t_stop; recordValues=["mass_s", "mass_v"], setup=false, reset=false, instantiate=false, saveat=tData)

# setup traing data
posData = fmi2GetSolutionValue(realSimData, "mass_s")
velData = fmi2GetSolutionValue(realSimData, "mass_v")

# loss function for training
function losssum()
    global problem, x0, posData
    solution = problem(x0)

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    velNet = fmi2GetSolutionState(solution, 2; isIndex=true)
    
    Flux.Losses.mse(posNet, posData) + Flux.Losses.mse(velNet, velData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb()
    global iterCB += 1
    global lastLoss

    if iterCB % 1 == 0
        loss = losssum()
        @info "Loss: $loss"

        # This test condition is weak, because when the FMU passes an event, the error might increase.  
        # ToDo: More intelligent testing condition.
        @test (loss < lastLoss*2.0) && (loss != lastLoss)
        lastLoss = loss
    end
end

vr = fmi2StringToValueReference(realFMU, "mass_m")

numStates = fmiGetNumberOfStates(realFMU)

# some NeuralFMU setups
nets = [] 

1 # default ME-NeuralFMU (learn dynamics and states, almost-neutral setup, parameter count << 100)
net = Chain(Dense( [1.0 0.0; 0.0 1.0] + rand(numStates,numStates)*0.01, zeros(numStates), identity),
            states ->  fmiEvaluateME(realFMU, states), 
            Dense( [1.0 0.0; 0.0 1.0] + rand(numStates,numStates)*0.01, zeros(numStates), identity))
push!(nets, net)

# 2 # default ME-NeuralFMU (learn dynamics)
net = Chain(states ->  fmiEvaluateME(realFMU, states), 
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 3 # default ME-NeuralFMU (learn states)
net = Chain(Dense(numStates, 16, identity),
            Dense(16, 16, identity),
            Dense(16, numStates),
            states -> fmiEvaluateME(realFMU, states))
push!(nets, net)

# 4 # default ME-NeuralFMU (learn dynamics and states)
net = Chain(Dense(numStates, 16, leakyrelu),
            Dense(16, 16, leakyrelu),
            Dense(16, numStates),
            states -> fmiEvaluateME(realFMU, states),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 5 # NeuralFMU with hard setting time to 0.0
net = Chain(states ->  fmiEvaluateME(realFMU, states), # not supported by this FMU:   states ->  fmiEvaluateME(realFMU, states, 0.0), 
            Dense(numStates, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 6 # NeuralFMU with additional getter 
getVRs = [fmi2StringToValueReference(realFMU, "mass_m")]
numGetVRs = length(getVRs)
net = Chain(states ->  fmiEvaluateME(realFMU, states, realFMU.components[end].t, fmi2ValueReference[], Real[], getVRs), 
            Dense(numStates+numGetVRs, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 7 # NeuralFMU with additional setter 
setVRs = [fmi2StringToValueReference(realFMU, "mass_m")]
numSetVRs = length(setVRs)
net = Chain(states ->  fmiEvaluateME(realFMU, states, realFMU.components[end].t, setVRs, [1.1]), 
            Dense(numStates, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 8 # NeuralFMU with additional setter and getter
net = Chain(states ->  fmiEvaluateME(realFMU, states, realFMU.components[end].t, setVRs, [1.1], getVRs), 
            Dense(numStates+numGetVRs, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 9 # Empty NeuralFMU
net = Chain(states ->  fmiEvaluateME(realFMU, states))
push!(nets, net)

optim = ADAM(1e-4)
for i in 1:length(nets)
    @testset "Net setup #$i" begin
        global nets, problem, lastLoss, iterCB
        net = nets[i]
        problem = ME_NeuralFMU(realFMU, net, (t_start, t_stop), Tsit5(); saveat=tData, rootSearchInterpolationPoints=1000)
        
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
        lastLoss = losssum()
        @info "Start-Loss for net #$i: $lastLoss"

        # net #9 has no parameters -> skip it
        if i != 9 
            Flux.train!(losssum, p_net, Iterators.repeated((), 1), optim; cb=callb) 
        end

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
