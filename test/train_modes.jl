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
myFMU = fmiLoad("SpringFrictionPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"])
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

# setup traing data
posData = fmi2GetSolutionValue(realSimData, "mass.s")

# loss function for training
function losssum()
    global problem, x0, posData
    solution = problem(x0)

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    Flux.Losses.mse(posNet, posData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb()
    global iterCB += 1
    global lastLoss

    if iterCB % 25 == 0
        loss = losssum()
        @info "Loss: $loss"
        @test loss < lastLoss  
        lastLoss = loss
    end
end

vr = fmi2StringToValueReference(myFMU, "mass.m")

numStates = fmiGetNumberOfStates(myFMU)

# some NeuralFMU setups
nets = [] 

for handleEvents in [true, false]
    @testset "handleEvents: $handleEvents" begin
        for config in [FMU_EXECUTION_CONFIGURATION_RESET, FMU_EXECUTION_CONFIGURATION_NO_RESET, FMU_EXECUTION_CONFIGURATION_NO_FREEING]
            @testset "config: $config" begin
                
                global problem, lastLoss, iterCB

                myFMU.executionConfig = config
                myFMU.executionConfig.handleStateEvents = handleEvents
                myFMU.executionConfig.handleTimeEvents = handleEvents

                net = Chain(Dense(numStates, 16),
                    Dense(16, numStates),
                    states -> fmiEvaluateME(myFMU, states),
                    Dense(numStates, 16, tanh),
                    Dense(16, numStates))
                
                optim = ADAM(1e-4)
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
                lastLoss = losssum()
                lastInstCount = length(problem.fmu.components)

                Flux.train!(losssum, p_net, Iterators.repeated((), 50), optim; cb=callb)

                # check results
                solutionAfter = problem(x0)
                if solutionAfter.success
                    @test length(solutionAfter.states.t) == length(tData)
                    @test solutionAfter.states.t[1] == t_start
                    @test solutionAfter.states.t[end] == t_stop
                end

                # clean-up the dead components
                while length(problem.fmu.components) > 1 
                    fmiFreeInstance!(problem.fmu)
                end
            end
              
        end
    end
end

fmiUnload(realFMU)
fmiUnload(myFMU)
