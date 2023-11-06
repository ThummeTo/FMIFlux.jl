#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMIFlux.Flux

import Random 
Random.seed!(1234);

t_start = 0.0
t_step = 0.01
t_stop = 50.0
tData = t_start:t_step:t_stop

# generate training data
posData = cos.(tData.*3.0)*2.0
velData = sin.(tData.*3.0)
x0 = [0.5, 0.0]

# load FMU for NeuralFMU
fmu = fmiLoad("SpringPendulum1D", EXPORTINGTOOL, EXPORTINGVERSION; type=fmi2TypeModelExchange)

using FMI.FMIImport
using FMI.FMIImport.FMICore

# loss function for training
function losssum_single(p)
    global problem, x0, posData
    solution = problem(x0; p=p, showProgress=true, saveat=tData)

    if !solution.success
        return Inf 
    end

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    return Flux.Losses.mse(posNet, posData)
end

function losssum_multi(p)
    global problem, x0, posData
    solution = problem(x0; p=p, showProgress=true, saveat=tData)

    if !solution.success
        return [Inf, Inf]
    end

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    velNet = fmi2GetSolutionState(solution, 2; isIndex=true)
    
    return [Flux.Losses.mse(posNet, posData), Flux.Losses.mse(velNet, velData)]
end

# callback function for training
global iterCB = 0
global lastLoss = Inf
function callb(losssum, p)
    global iterCB += 1
    global lastLoss

    if iterCB % 5 == 0
        newloss = losssum(p[1])
        loss = (length(newloss) > 1 ? sum(newloss) : newloss)
        @info "[$(iterCB)] Loss: $loss"
        @test loss < lastLoss  
        lastLoss = loss
    end
end

numStates = fmiGetNumberOfStates(fmu)

c1 = CacheLayer()
c2 = CacheRetrieveLayer(c1)

# the "Chain" for training
net = Chain(x -> fmu(;x=x, dx_refs=:all),
            dx -> c1(dx),
            Dense(numStates, 12, tanh),
            Dense(12, 1, identity),
            dx -> c2([1], dx[1], []))

solver = Tsit5()
problem = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver; saveat=tData)
@test problem != nothing

solutionBefore = problem(x0)

# train it ...
p_net = Flux.params(problem)

iterCB = 0

# single objective
lastLoss = losssum_single(p_net[1])
optim = Adam(1e-3)
FMIFlux.train!(losssum_single, p_net, Iterators.repeated((), NUMSTEPS), optim; cb=()->callb(losssum_single, p_net), gradient=GRADIENT)

# multi objective
# lastLoss = sum(losssum_multi(p_net[1]))
# optim = Adam(1e-3)
# FMIFlux.train!(losssum_multi,  p_net, Iterators.repeated((), NUMSTEPS), optim; cb=()->callb(losssum_multi,  p_net), gradient=GRADIENT, multiObjective=true)

# check results
solutionAfter = problem(x0)
@test solutionAfter.success
@test length(solutionAfter.states.t) == length(tData)
@test solutionAfter.states.t[1] == t_start
@test solutionAfter.states.t[end] == t_stop

@test length(fmu.components) <= 1

fmi2Unload(fmu)
