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
posData = sin.(tData.*3.0)*2.0
velData = cos.(tData.*3.0)*6.0
accData = sin.(tData.*3.0)*-18.0
x0 = [0.5, 0.0]

# load FMU for NeuralFMU
fmu = fmi2Load("SpringFrictionPendulum1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:ME)

# loss function for training
function losssum(p)
    global problem, x0, posData
    solution = problem(x0; p=p, saveat=tData)

    if !solution.success
        return Inf 
    end

    #posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    velNet = fmi2GetSolutionState(solution, 2; isIndex=true)
    
    return Flux.Losses.mse(velNet, velData) # Flux.Losses.mse(posNet, posData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb(p)
    global iterCB += 1
    global lastLoss

    if iterCB % 5 == 0
        loss = losssum(p[1])
        @info "[$(iterCB)] Loss: $loss"
        @test loss < lastLoss  
        lastLoss = loss
    end
end

vr = fmi2StringToValueReference(fmu, "mass.m")

numStates = length(fmu.modelDescription.stateValueReferences)

# some NeuralFMU setups
nets = [] 

global comp
comp = nothing
for handleEvents in [true, false]
    @testset "handleEvents: $handleEvents" begin
        for config in [FMU2_EXECUTION_CONFIGURATION_NO_RESET, FMU2_EXECUTION_CONFIGURATION_RESET, FMU2_EXECUTION_CONFIGURATION_NO_FREEING]
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
                #     comp, _ = prepareSolveFMU(fmu, comp, :ME, instantiate, nothing, nothing, nothing, nothing, nothing, t_start, t_stop, nothing; x0=x0, handleEvents=FMIFlux.handleEvents, cleanup=true)
                # end

                c1 = CacheLayer()
                c2 = CacheRetrieveLayer(c1)

                net = Chain(states -> fmu(;x=states, dx_refs=:all),
                            dx -> c1(dx),
                            Dense(numStates, 16, tanh),
                            Dense(16, 1, identity),
                            dx -> c2([1], dx[1], []) )
                
                optim = Adam(1e-8)
                solver = Tsit5()

                problem = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver)
                @test problem != nothing
                
                solutionBefore = problem(x0; saveat=tData)
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

                @info "[ $(iterCB)] Loss: $lastLoss"
                FMIFlux.train!(losssum, p_net, Iterators.repeated((), NUMSTEPS), optim; cb=()->callb(p_net), gradient=GRADIENT)

                # check results
                solutionAfter = problem(x0; saveat=tData)
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

fmi2Unload(fmu)
