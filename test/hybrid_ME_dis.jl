#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using DifferentialEquations: Tsit5, Rosenbrock23

import Random 
Random.seed!(5678);

t_start = 0.0
t_step = 0.1
t_stop = 3.0
tData = t_start:t_step:t_stop

# generate training data
posData = sin.(tData.*3.0)*2.0
velData = cos.(tData.*3.0)*6.0
accData = sin.(tData.*3.0)*-18.0
x0 = [0.5, 0.0]

# loss function for training
function losssum(p)
    global problem, x0, posData
    solution = problem(x0; p=p, saveat=tData)

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
        @test loss <= lastLoss
        lastLoss = loss
    end
end

vr = fmi2StringToValueReference(fmu, "g")

numStates = length(fmu.modelDescription.stateValueReferences)

# some NeuralFMU setups
nets = []

c1 = CacheLayer()
c2 = CacheRetrieveLayer(c1)
c3 = CacheLayer()
c4 = CacheRetrieveLayer(c3)

init = Flux.glorot_uniform
getVRs = [fmi2StringToValueReference(fmu, "h")]
y = zeros(fmi2Real, length(getVRs))
numGetVRs = length(getVRs)
setVRs = [fmi2StringToValueReference(fmu, "v")]
numSetVRs = length(setVRs)

# 1. default ME-NeuralFMU (learn dynamics and states, almost-neutral setup, parameter count << 100)
net = Chain(x -> c1(x),
            Dense(numStates, 1, identity; init=init),
            x -> c2([], x[1], [1]),
            x -> fmu(;x=x, dx_refs=:all), 
            x -> c3(x),
            Dense(numStates, 1, identity; init=init),
            x -> c4([1], x[1], []))
push!(nets, net)

# 2. default ME-NeuralFMU (learn dynamics)
net = Chain(x -> fmu(;x=x, dx_refs=:all), 
            x -> c1(x),
            Dense(numStates, 16, tanh; init=init),
            Dense(16, 16, tanh; init=init),
            Dense(16, 1, identity; init=init),
            x -> c2([1], x[1], []))
push!(nets, net)

# 3. default ME-NeuralFMU (learn states)
net = Chain(x -> c1(x),
            Dense(numStates, 16, tanh, init=init),
            Dense(16, 16, tanh, init=init),
            Dense(16, 1, identity, init=init),
            x -> c2([], x[1], [1]),
            x -> fmu(;x=x, dx_refs=:all))
push!(nets, net)

# 4. default ME-NeuralFMU (learn dynamics and states)
net = Chain(x -> c1(x),
            Dense(numStates, 16, tanh; init=init),
            Dense(16, 1, identity; init=init),
            x -> c2([], x[1], [1]),
            x -> fmu(;x=x, dx_refs=:all), 
            x -> c3(x),
            Dense(numStates, 16, tanh, init=init),
            Dense(16, 16, tanh, init=init),
            Dense(16, 1, identity, init=init),
            x -> c4([1], x[1], []))
push!(nets, net)

# 5. NeuralFMU with hard setting time to 0.0
net = Chain(states -> fmu(;x=states, t=0.0, dx_refs=:all),
            x -> c1(x),
            Dense(numStates, 8, tanh; init=init),
            Dense(8, 16, tanh; init=init),
            Dense(16, 1, identity; init=init),
            x -> c2([1], x[1], []))
push!(nets, net)

# 6. NeuralFMU with additional getter 
net = Chain(x -> fmu(;x=x, y_refs=getVRs, y=y, dx_refs=:all), 
            x -> c1(x),
            Dense(numStates+numGetVRs, 8, tanh; init=init),
            Dense(8, 16, tanh; init=init),
            Dense(16, 1, identity; init=init),
            x -> c2([2], x[1], []))
push!(nets, net)

# 7. NeuralFMU with additional setter 
net = Chain(x -> fmu(;x=x, u_refs=setVRs, u=[1.1], dx_refs=:all), 
            x -> c1(x),
            Dense(numStates, 8, tanh; init=init),
            Dense(8, 16, tanh; init=init),
            Dense(16, 1, identity; init=init),
            x -> c2([1], x[1], []))
push!(nets, net)

# 8. NeuralFMU with additional setter and getter
net = Chain(x -> fmu(;x=x, u_refs=setVRs, u=[1.1], y_refs=getVRs, y=y, dx_refs=:all),
            x -> c1(x),
            Dense(numStates+numGetVRs, 8, tanh; init=init),
            Dense(8, 16, tanh; init=init),
            Dense(16, 1, identity; init=init),
            x -> c2([2], x[1], []))
push!(nets, net)

# 9. an empty NeuralFMU (this does only make sense for debugging)
net = Chain(x -> fmu(x=x, dx_refs=:all))
push!(nets, net)

for i in 1:length(nets)
    @testset "Net setup $(i)/$(length(nets))" begin
        global nets, problem, lastLoss, iterCB

        optim = Adam(1e-6)
        solver = Tsit5()

        net = nets[i]
        problem = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver) 

        # train it ...
        p_net = Flux.params(problem)
        
        @test problem !== nothing

        solutionBefore = problem(x0; p=p_net[1], saveat=tData)
        if solutionBefore.success
            @test length(solutionBefore.states.t) == length(tData)
            @test solutionBefore.states.t[1] == t_start
            @test solutionBefore.states.t[end] == t_stop
        end

        iterCB = 0
        lastLoss = losssum(p_net[1])
        @info "[ $(iterCB)] Loss: $lastLoss"
        FMIFlux.train!(losssum, p_net, Iterators.repeated((), NUMSTEPS), optim; cb=()->callb(p_net), gradient=GRADIENT)

        # check results
        solutionAfter = problem(x0; p=p_net[1], saveat=tData)
        if solutionAfter.success
            @test length(solutionAfter.states.t) == length(tData)
            @test solutionAfter.states.t[1] == t_start
            @test solutionAfter.states.t[end] == t_stop
        end
    end
end

@test length(fmu.components) <= 1

fmi2Unload(fmu)
