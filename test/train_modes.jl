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
_, realSimData = fmiSimulateCS(realFMU, t_start, t_stop; recordValues=["mass.s", "mass.v"], setup=false, reset=false, saveat=tData)

# load FMU for NeuralFMU
myFMU = fmiLoad("SpringFrictionPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"])
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

# setup traing data
posData = collect(data[1] for data in realSimData.saveval)

# loss function for training
function losssum()
    global problem, x0, posData
    solution = problem(x0)

    posNet = collect(data[1] for data in solution.u)
    
    Flux.Losses.mse(posNet, posData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb()
    global iterCB += 1
    global lastLoss

    if iterCB % 30 == 0
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

# net
net = Chain(Dense(numStates, 16, leakyrelu),
            Dense(16, 16, leakyrelu),
            Dense(16, numStates),
            states -> fmiEvaluateME(myFMU, states),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))

for handleEvents in [true, false]
    @testset "handleEvents: $handleEvents" begin
        for instantiate in [true, false]
            @testset "instantiate: $instantiate" begin
                for reset in [true, false]
                    @testset "reset: $reset" begin
                        for freeInstance in [true, false]
                            @testset "freeInstance: $freeInstance" begin
                                global problem, lastLoss, iterCB

                                config = NeuralFMU_TrainingModeConfig()
                                config.handleStateEvents = handleEvents
                                config.handleTimeEvents = handleEvents
                                config.instantiate = instantiate
                                config.reset = reset 
                                config.freeInstance = freeInstance
                            
                                optim = ADAM(1e-4)
                                problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(); saveat=tData)
                                @test problem != nothing

                                problem.trainingConfig = config
                                
                                solutionBefore = problem(x0)
                                @test length(solutionBefore.t) == length(tData)
                                @test solutionBefore.t[1] == t_start
                                @test solutionBefore.t[end] == t_stop

                                # train it ...
                                p_net = Flux.params(problem)

                                iterCB = 0
                                lastLoss = losssum()
                                lastInstCount = length(problem.fmu.components)

                                Flux.train!(losssum, p_net, Iterators.repeated((), 60), optim; cb=callb)

                                if !freeInstance
                                    if instantiate
                                        @test (length(problem.fmu.components) - lastInstCount) >= 60 # more than 60 because forward diff multiple runs
                                    else
                                        @test (length(problem.fmu.components) == lastInstCount)
                                    end
                                end

                                # check results
                                solutionAfter = problem(x0)
                                @test length(solutionAfter.t) == length(tData)
                                @test solutionAfter.t[1] == t_start
                                @test solutionAfter.t[end] == t_stop
                            end
                        end
                    end
                end
            end
        end
    end
end

fmiUnload(realFMU)
fmiUnload(myFMU)
