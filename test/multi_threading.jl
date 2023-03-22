#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
import Flux
using DifferentialEquations: Tsit5, Rosenbrock23

import Random 
Random.seed!(5678);

t_start = 0.0
t_step = 0.1
t_stop = 3.0
tData = t_start:t_step:t_stop

# generate training data
realFMU = fmiLoad("SpringFrictionPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"]; type=:ME)
pdict = Dict("mass.m" => 1.3)
realSimData = fmiSimulate(realFMU, (t_start, t_stop); parameters=pdict, recordValues=["mass.s", "mass.v"], saveat=tData)
x0 = collect(realSimData.values.saveval[1])
@test x0 == [0.5, 0.0]

# load FMU for training
realFMU = fmiLoad("SpringFrictionPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"]; type=fmi2TypeModelExchange)

# setup traing data
velData = fmi2GetSolutionValue(realSimData, "mass.v")

# loss function for training
function losssum(p)
    global problem, x0, posData
    solution = problem(x0; p=p)

    if !solution.success
        return Inf 
    end

    # posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    velNet = fmi2GetSolutionState(solution, 2; isIndex=true)
    
    return FMIFlux.Losses.mse(velNet, velData) # Flux.Losses.mse(posNet, posData)
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

        # This test condition is not good, because when the FMU passes an event, the error might increase.  
        @test (loss < lastLoss) && (loss != lastLoss)
        lastLoss = loss
    end
end

vr = fmi2StringToValueReference(realFMU, "mass.m")

numStates = fmiGetNumberOfStates(realFMU)

# some NeuralFMU setups
nets = []

c1 = CacheLayer()
c2 = CacheRetrieveLayer(c1)
c3 = CacheLayer()
c4 = CacheRetrieveLayer(c3)

# 1. Discontinuous ME-NeuralFMU (learn dynamics and states)
net = Chain(x -> c1(x),
            Dense(numStates, 16, identity; init=FMIFlux.identity_init_64),
            Dense(16, numStates, identity; init=FMIFlux.identity_init_64),
            x -> c2([1], x[2], []),
            x -> realFMU(;x=x), 
            x -> c3(x),
            Dense(numStates, 16, identity, init=FMIFlux.identity_init_64),
            Dense(16, 16, identity, init=FMIFlux.identity_init_64),
            Dense(16, numStates, identity, init=FMIFlux.identity_init_64),
            x -> c4([1], x[2], []))
push!(nets, net)

for i in 1:length(nets)
    @testset "Net setup $(i)/$(length(nets))" begin
        global nets, problem, lastLoss, iterCB

        net = nets[i]
        solver = Tsit5()
        problem = ME_NeuralFMU(realFMU, net, (t_start, t_stop), solver; saveat=tData)
        
        @test problem !== nothing

        solutionBefore = problem(x0)
        if solutionBefore.success
            @test length(solutionBefore.states.t) == length(tData)
            @test solutionBefore.states.t[1] == t_start
            @test solutionBefore.states.t[end] == t_stop
        end

        # train it ...
        p_net = Flux.params(problem)
        p_start = copy(p_net[1])

        iterCB = 0
        lastLoss = losssum(p_net[1])
        startLoss = lastLoss
        @info "[ $(iterCB)] Loss: $lastLoss"

        p_net[1][:] = p_start[:]
        lastLoss = startLoss
        st = time()
        optim = Adam(1e-4)
        FMIFlux.train!(losssum, p_net, Iterators.repeated((), parse(Int, ENV["NUMSTEPS"])), optim; cb=()->callb(p_net), multiThreading=false)
        dt = round(time()-st; digits=1)
        @info "Training time single threaded (not pre-compiled): $(dt)s"

        p_net[1][:] = p_start[:]
        lastLoss = startLoss
        st = time()
        optim = Adam(1e-4)
        FMIFlux.train!(losssum, p_net, Iterators.repeated((), parse(Int, ENV["NUMSTEPS"])), optim; cb=()->callb(p_net), multiThreading=false)
        dt = round(time()-st; digits=1)
        @info "Training time single threaded (pre-compiled): $(dt)s"

        p_net[1][:] = p_start[:]
        lastLoss = startLoss
        st = time()
        optim = Adam(1e-4)
        FMIFlux.train!(losssum, p_net, Iterators.repeated((), parse(Int, ENV["NUMSTEPS"])), optim; cb=()->callb(p_net), multiThreading=true)
        dt = round(time()-st; digits=1)
        @info "Training time multi threaded (not pre-compiled): $(dt)s"

        p_net[1][:] = p_start[:]
        lastLoss = startLoss
        st = time()
        optim = Adam(1e-4)
        FMIFlux.train!(losssum, p_net, Iterators.repeated((), parse(Int, ENV["NUMSTEPS"])), optim; cb=()->callb(p_net), multiThreading=true)
        dt = round(time()-st; digits=1)
        @info "Training time multi threaded (pre-compiled): $(dt)s"

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
