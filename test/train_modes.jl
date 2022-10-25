#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using DifferentialEquations: Tsit5, Rosenbrock23

import Random 
Random.seed!(5678);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
realFMU = fmiLoad("SpringFrictionPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"])
pdict = Dict("mass.m" => 1.3)
realSimData = fmiSimulate(realFMU, t_start, t_stop; parameters=pdict, recordValues=["mass.s", "mass.v"], saveat=tData)
x0 = collect(realSimData.values.saveval[1])
@test x0 == [0.5, 0.0]

# load FMU for NeuralFMU
myFMU = fmiLoad("SpringFrictionPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"]; type=:ME)

# setup traing data
posData = fmi2GetSolutionValue(realSimData, "mass.s")
velData = fmi2GetSolutionValue(realSimData, "mass.v")

# loss function for training
function losssum(p)
    global problem, x0, posData
    solution = problem(x0; p=p)

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

    if iterCB % 10 == 0
        loss = losssum(p[1])
        @info "[$(iterCB)] Loss: $loss"
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
        for config in [FMU2_EXECUTION_CONFIGURATION_RESET, FMU2_EXECUTION_CONFIGURATION_NO_RESET, FMU2_EXECUTION_CONFIGURATION_NO_FREEING]
            @testset "config: $config" begin
                
                global problem, lastLoss, iterCB

                myFMU.executionConfig = config
                myFMU.executionConfig.handleStateEvents = handleEvents
                myFMU.executionConfig.handleTimeEvents = handleEvents
                myFMU.executionConfig.externalCallbacks = true
                myFMU.executionConfig.loggingOn = true
                myFMU.executionConfig.assertOnError = true
                myFMU.executionConfig.assertOnWarning = true

                @info "handleEvents: $(handleEvents) | instantiate: $(myFMU.executionConfig.instantiate) | reset: $(myFMU.executionConfig.reset) | freeInstance: $(myFMU.executionConfig.freeInstance)"

                if myFMU.executionConfig.instantiate == false 
                    #@info "instantiate = false, instantiating..."

                    comp = FMI.fmi2Instantiate!(myFMU; loggingOn=false)
                    FMI.fmi2SetupExperiment(comp, t_start, t_stop)
                    FMI.fmi2EnterInitializationMode(comp)
                    FMI.fmi2ExitInitializationMode(comp)

                    FMIFlux.handleEvents(comp)

                    FMI.fmi2Terminate(comp)
                end

                c1 = CacheLayer()
                c2 = CacheRetrieveLayer(c1)
                c3 = CacheLayer()
                c4 = CacheRetrieveLayer(c3)
                net = Chain(x -> c1(x),
                            Dense(numStates, numStates, identity; init=Flux.identity_init),
                            x -> c2([1], x[2], []),
                            states -> myFMU(;x=states)[2],
                            x -> c3(x),
                            Dense(numStates, 16, identity; init=Flux.identity_init),
                            Dense(16, numStates, identity; init=Flux.identity_init),
                            x -> c4([1], x[2], []) )
                
                optim = Adam(1e-6)
                problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Rosenbrock23(autodiff=false); saveat=tData)
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
                lastInstCount = length(problem.fmu.components)

                @info "[$(iterCB)] Loss: $lastLoss"
                FMIFlux.train!(losssum, p_net, Iterators.repeated((), 30), optim; cb=()->callb(p_net))

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
