#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using DifferentialEquations: Tsit5, Rosenbrock23
import FMIFlux.FMIImport: fmi2FreeInstance!

import Random 
Random.seed!(5678);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
posData, velData, accData = syntTrainingData(tData)

# load FMU for NeuralFMU
fmu = loadFMU("SpringFrictionPendulum1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:ME)

# loss function for training
losssum = function(p)
    global problem, X0, posData
    solution = problem(X0; p=p, saveat=tData)

    if !solution.success
        return Inf 
    end

    #posNet = getState(solution, 1; isIndex=true)
    velNet = getState(solution, 2; isIndex=true)
    
    return Flux.Losses.mse(velNet, velData) # Flux.Losses.mse(posNet, posData)
end

vr = stringToValueReference(fmu, "mass.m")

numStates = length(fmu.modelDescription.stateValueReferences)

# some NeuralFMU setups
nets = [] 

global comp
comp = nothing
for handleEvents in [true, false]
    @testset "handleEvents: $handleEvents" begin
        for config in FMU_EXECUTION_CONFIGURATIONS
            @testset "config: $config" begin
                
                global problem, lastLoss, iterCB, comp

                fmu.executionConfig = config
                fmu.executionConfig.handleStateEvents = handleEvents
                fmu.executionConfig.handleTimeEvents = handleEvents
                fmu.executionConfig.externalCallbacks = true
                fmu.executionConfig.loggingOn = true
                fmu.executionConfig.assertOnError = true
                fmu.executionConfig.assertOnWarning = true

                @info "handleEvents: $(handleEvents) | instantiate: $(fmu.executionConfig.instantiate) | reset: $(fmu.executionConfig.reset)  | terminate: $(fmu.executionConfig.terminate) | freeInstance: $(fmu.executionConfig.freeInstance)"

                # if fmu.executionConfig.instantiate == false 
                #     @info "instantiate = false, instantiating..."
                #     instantiate = true
                #     comp, _ = prepareSolveFMU(fmu, comp, :ME, instantiate, nothing, nothing, nothing, nothing, nothing, t_start, t_stop, nothing; x0=X0, handleEvents=FMIFlux.handleEvents, cleanup=true)
                # end

                c1 = CacheLayer()
                c2 = CacheRetrieveLayer(c1)

                net = Chain(states -> fmu(;x=states, dx_refs=:all),
                            dx -> c1(dx),
                            Dense(numStates, 16, tanh),
                            Dense(16, 1, identity),
                            dx -> c2(1, dx[1]) )
                
                optim = OPTIMISER(ETA)
                solver = Tsit5()

                problem = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver)
                @test problem != nothing
                
                solutionBefore = problem(X0; saveat=tData)
                if solutionBefore.success
                    @test length(solutionBefore.states.t) == length(tData)
                    @test solutionBefore.states.t[1] == t_start
                    @test solutionBefore.states.t[end] == t_stop
                end

                # train it ...
                p_net = Flux.params(problem)

                iterCB = 0
                lastLoss = losssum(p_net[1])
                lastInstCount = length(problem.fmu.components)
                @info "Start-Loss for net: $lastLoss"

                lossBefore = losssum(p_net[1])
                FMIFlux.train!(losssum, problem, Iterators.repeated((), NUMSTEPS), optim; gradient=GRADIENT)
                lossAfter = losssum(p_net[1])

                @test lossAfter < lossBefore

                # check results
                solutionAfter = problem(X0; saveat=tData)
                if solutionAfter.success
                    @test length(solutionAfter.states.t) == length(tData)
                    @test solutionAfter.states.t[1] == t_start
                    @test solutionAfter.states.t[end] == t_stop
                end

                # this is not possible, because some pullbacks are evaluated after simulation end
                while length(problem.fmu.components) > 1 
                    fmi2FreeInstance!(problem.fmu.components[end])
                end

            end
              
        end
    end
end

unloadFMU(fmu)
