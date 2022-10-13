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
realSimData = fmiSimulate(realFMU, t_start, t_stop; recordValues=["mass_s", "mass_v"], saveat=tData)
x0 = collect(realSimData.values.saveval[1])
@test x0 == [1.0, 0.0]

# setup traing data
velData = fmi2GetSolutionValue(realSimData, "mass_v")

# loss function for training
function losssum(p)
    global problem, x0, posData
    solution = problem(x0; p=p)

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

    if iterCB % 10 == 0
        loss = losssum(p[1])
        @info "Loss: $loss"

        # This test condition is not good, because when the FMU passes an event, the error might increase.  
        @test (loss < lastLoss) && (loss != lastLoss)
        lastLoss = loss
    end
end

vr = fmi2StringToValueReference(realFMU, "mass_m")

numStates = fmiGetNumberOfStates(realFMU)

# some NeuralFMU setups
nets = [] 

# 1. default ME-NeuralFMU (learn dynamics and states, almost-neutral setup, parameter count << 100)
net = Chain(Dense(numStates, numStates, tanh; init=Flux.identity_init),
            states -> fmiEvaluateME(realFMU, states), 
            Dense(numStates, numStates, identity; init=Flux.identity_init))
push!(nets, net)

# 2. default ME-NeuralFMU (learn dynamics)
net = Chain(states -> fmiEvaluateME(realFMU, states), 
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 3. default ME-NeuralFMU (learn states)
c1 = CacheLayer()
c2 = CacheRetrieveLayer(c1)
function eval(x)
    y, dx = realFMU(;x=x)
    return dx
end
net = Chain(x -> c1(x),
            Dense(numStates, 16, identity, init=Flux.identity_init),
            Dense(16, 16, identity, init=Flux.identity_init),
            Dense(16, 1, identity, init=Flux.identity_init),
            x -> c2([1], x, []),
            x -> eval(x))
push!(nets, net)

# 4. default ME-NeuralFMU (learn dynamics and states)
net = Chain(Dense(numStates, 16, leakyrelu),
            Dense(16, 16, leakyrelu),
            Dense(16, numStates),
            states -> fmiEvaluateME(realFMU, states),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 5. NeuralFMU with hard setting time to 0.0
net = Chain(states -> realFMU(;x=states, t=0.0)[2],
            x -> c1(x),
            Dense(numStates, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, 1),
            x -> c2([1], x, []))
push!(nets, net)

# 6. NeuralFMU with additional getter 
getVRs = [fmi2StringToValueReference(realFMU, "mass_s")]
numGetVRs = length(getVRs)
function eval(x)
    y, dx = realFMU(;x=x, y_refs=getVRs)
    return [y..., dx...]
end
net = Chain(x -> eval(x), 
            x -> c1(x),
            Dense(numStates+numGetVRs, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, 1),
            x -> c2([2], x, []))
push!(nets, net)

# 7. NeuralFMU with additional setter 
setVRs = [fmi2StringToValueReference(realFMU, "mass_m")]
numSetVRs = length(setVRs)
net = Chain(states ->  fmiEvaluateME(realFMU, states, realFMU.components[end].t, setVRs, [1.1]), 
            Dense(numStates, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

# 8. NeuralFMU with additional setter and getter
function eval(x)
    y, dx = realFMU(;x=x, u_refs=setVRs, u=[1.1], y_refs=getVRs)
    return [y..., dx...]
end
net = Chain(x -> eval(x),
            Dense(numStates+numGetVRs, 8, tanh),
            Dense(8, 16, tanh),
            Dense(16, numStates))
push!(nets, net)

optim = Adam(1e-6)
for i in 1:length(nets)
    @testset "Net setup #$i" begin
        global nets, problem, lastLoss, iterCB

        net = nets[i]
        problem = ME_NeuralFMU(realFMU, net, (t_start, t_stop), Tsit5(); saveat=tData)
        
        @test problem !== nothing

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
