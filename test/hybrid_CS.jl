#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux

import Random 
Random.seed!(1234);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
posData, velData, accData = syntTrainingData(tData)

fmu = loadFMU("SpringPendulumExtForce1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:CS)

# sine(t) as external force
extForce = function(t)
    return [sin(t)]
end

# loss function for training
losssum = function(p)
    solution = problem(extForce, t_step; p=p)

    if !solution.success
        return Inf 
    end

    accNet = getValue(solution, 1; isIndex=true)

    Flux.Losses.mse(accNet, accData)
end

# NeuralFMU setup
numInputs = length(fmu.modelDescription.inputValueReferences)
numOutputs = length(fmu.modelDescription.outputValueReferences)

net = Chain(u -> fmu(;u_refs=fmu.modelDescription.inputValueReferences, u=u, y_refs=fmu.modelDescription.outputValueReferences),
            Dense(numOutputs, 16, tanh; init=Flux.identity_init),
            Dense(16, 16, tanh; init=Flux.identity_init),
            Dense(16, numOutputs; init=Flux.identity_init))

problem = CS_NeuralFMU(fmu, net, (t_start, t_stop))
@test problem != nothing

# train it ...
p_net = Flux.params(problem)

lossBefore = losssum(p_net[1])
optim = OPTIMISER(ETA)

FMIFlux.train!(losssum, problem, Iterators.repeated((), NUMSTEPS), optim; gradient=GRADIENT)

lossAfter = losssum(p_net[1])
@test lossAfter < lossBefore

@test length(fmu.components) <= 1

unloadFMU(fmu)
