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
realFMU = fmiLoad("SpringFrictionPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"]; type=fmi2TypeCoSimulation)
realSimData = fmiSimulateCS(realFMU, (t_start, t_stop); recordValues=["mass.s", "mass.v"], saveat=tData)
x0 = collect(realSimData.values.saveval[1])
@test x0 == [0.5, 0.0]

# load FMU for NeuralFMU
myFMU = fmiLoad("SpringPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"]; type=fmi2TypeModelExchange)

# setup traing data
velData = fmi2GetSolutionValue(realSimData, "mass.v")

# loss function for training
function losssum(p)
    global problem, x0, posData
    solution = problem(x0; p=p, showProgress=true)

    if !solution.success
        return Inf 
    end

    # posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
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

numStates = fmiGetNumberOfStates(myFMU)

# some NeuralFMU setups
nets = [] 

c1 = CacheLayer()
c2 = CacheRetrieveLayer(c1)
c3 = CacheLayer()
c4 = CacheRetrieveLayer(c3)

# 1. default ME-NeuralFMU (learn dynamics and states, almost-neutral setup, parameter count << 100)
net = Chain(c1,
            Dense(numStates, numStates, identity; init=Flux.identity_init),
            x -> c2([1], x[2], []),
            x -> myFMU(;x=x), 
            c3,
            Dense(numStates, 1, identity; init=Flux.identity_init),
            x -> c4([1], x[1], []))
push!(nets, net)

# 2. default ME-NeuralFMU (learn dynamics)
net = Chain(x -> myFMU(;x=x), 
            x -> c1(x),
            Dense(numStates, 16, identity; init=Flux.identity_init),
            Dense(16, 16, identity; init=Flux.identity_init),
            Dense(16, 1, identity; init=Flux.identity_init),
            x -> c2([1], x[1], []))
push!(nets, net)

# 3. default ME-NeuralFMU (learn states)
net = Chain(x -> c1(x),
            Dense(numStates, 16, identity, init=Flux.identity_init),
            Dense(16, 16, identity, init=Flux.identity_init),
            Dense(16, numStates, identity, init=Flux.identity_init),
            x -> c2([1], x[2], []),
            x -> myFMU(;x=x))
push!(nets, net)

# 4. default ME-NeuralFMU (learn dynamics and states)
net = Chain(x -> c1(x),
            Dense(numStates, 16, identity; init=Flux.identity_init),
            Dense(16, numStates, identity; init=Flux.identity_init),
            x -> c2([1], x[2], []),
            x -> myFMU(;x=x), 
            x -> c3(x),
            Dense(numStates, 16, identity, init=Flux.identity_init),
            Dense(16, 16, identity, init=Flux.identity_init),
            Dense(16, 1, identity, init=Flux.identity_init),
            x -> c4([1], x[1], []))
push!(nets, net)

# 5. NeuralFMU with hard setting time to 0.0
net = Chain(states -> myFMU(;x=states, t=0.0),
            x -> c1(x),
            Dense(numStates, 8, identity; init=Flux.identity_init),
            Dense(8, 16, identity; init=Flux.identity_init),
            Dense(16, 1, identity; init=Flux.identity_init),
            x -> c2([1], x[1], []))
push!(nets, net)

# 6. NeuralFMU with additional getter 
getVRs = [fmi2StringToValueReference(myFMU, "mass.s")]
numGetVRs = length(getVRs)
net = Chain(x -> myFMU(;x=x, y_refs=getVRs), 
            x -> c1(x),
            Dense(numStates+numGetVRs, 8, identity; init=Flux.identity_init),
            Dense(8, 16, identity; init=Flux.identity_init),
            Dense(16, 1, identity; init=Flux.identity_init),
            x -> c2([2], x[1], []))
push!(nets, net)

# 7. NeuralFMU with additional setter 
setVRs = [fmi2StringToValueReference(myFMU, "mass.m")]
numSetVRs = length(setVRs)
net = Chain(x -> myFMU(;x=x, u_refs=setVRs, u=[1.1]), 
            x -> c1(x),
            Dense(numStates, 8, identity; init=Flux.identity_init),
            Dense(8, 16, identity; init=Flux.identity_init),
            Dense(16, 1, identity; init=Flux.identity_init),
            x -> c2([1], x[1], []))
push!(nets, net)

# 8. NeuralFMU with additional setter and getter
net = Chain(x -> myFMU(;x=x, u_refs=setVRs, u=[1.1], y_refs=getVRs),
            x -> c1(x),
            Dense(numStates+numGetVRs, 8, identity; init=Flux.identity_init),
            Dense(8, 16, identity; init=Flux.identity_init),
            Dense(16, 1, identity; init=Flux.identity_init),
            x -> c2([2], x[1], []))
push!(nets, net)

# 9. an empty NeuralFMU (this does only make sense for debugging)
net = Chain(x -> myFMU(;x=x))
push!(nets, net)

for i in 1:length(nets)
    @testset "Net setup #$i" begin
        global nets, problem, lastLoss, iterCB

        optim = Adam(1e-4)
        solver = Tsit5()

        net = nets[i]
        problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), solver; saveat=tData)
        @test problem != nothing

        solutionBefore = problem(x0)
        if solutionBefore.success
            @test length(solutionBefore.states.t) == length(tData)
            @test solutionBefore.states.t[1] == t_start
            @test solutionBefore.states.t[end] == t_stop
        end

        # train it ...
        p_net = Flux.params(problem)
        @test length(p_net) == 1

        iterCB = 0
        lastLoss = losssum(p_net[1])
        @info "Start-Loss for net #$i: $lastLoss"
        FMIFlux.train!(losssum, p_net, Iterators.repeated((), parse(Int, ENV["NUMSTEPS"])), optim; cb=()->callb(p_net))

        # check results
        solutionAfter = problem(x0)
        if solutionAfter.success
            @test length(solutionAfter.states.t) == length(tData)
            @test solutionAfter.states.t[1] == t_start
            @test solutionAfter.states.t[end] == t_stop
        end
    end
end

@test length(myFMU.components) <= 1

fmiUnload(realFMU)
fmiUnload(myFMU)
