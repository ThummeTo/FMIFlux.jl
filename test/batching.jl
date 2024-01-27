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
posData, velData, accData = syntTrainingData(tData)

# load FMU for NeuralFMU
fmu = fmi2Load("SpringPendulum1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:ME)

using FMIFlux.FMIImport
using FMIFlux.FMIImport.FMICore

# loss function for training
losssum_single = function(p)
    global problem, X0, posData
    solution = problem(X0; p=p, showProgress=true, saveat=tData)

    if !solution.success
        return Inf 
    end

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    return Flux.Losses.mse(posNet, posData)
end

losssum_multi = function(p)
    global problem, X0, posData
    solution = problem(X0; p=p, showProgress=true, saveat=tData)

    if !solution.success
        return [Inf, Inf]
    end

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    velNet = fmi2GetSolutionState(solution, 2; isIndex=true)
    
    return [Flux.Losses.mse(posNet, posData), Flux.Losses.mse(velNet, velData)]
end

numStates = length(fmu.modelDescription.stateValueReferences)

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

# before
p_net = Flux.params(problem)
lossBefore = losssum_single(p_net[1])

# single objective
optim = OPTIMISER(ETA)
FMIFlux.train!(losssum_single, problem, Iterators.repeated((), NUMSTEPS), optim; gradient=GRADIENT)

# multi objective
# lastLoss = sum(losssum_multi(p_net[1]))
# optim = OPTIMISER(ETA)
# FMIFlux.train!(losssum_multi,  problem, Iterators.repeated((), NUMSTEPS), optim; gradient=GRADIENT, multiObjective=true)

# after
lossAfter = losssum_single(p_net[1])
@test lossAfter < lossBefore

@test length(fmu.components) <= 1

fmi2Unload(fmu)
