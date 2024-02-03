#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using DifferentialEquations: Tsit5

import Random 
Random.seed!(1234);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
posData, velData, accData = syntTrainingData(tData)

# setup traing data
extForce = function(t)
    return [sin(t), cos(t)]
end

# loss function for training
losssum = function(p)
    solution = problem(extForce, t_step; p=p)

    if !solution.success
        return Inf 
    end

    accNet = fmi2GetSolutionValue(solution, 1; isIndex=true)

    FMIFlux.Losses.mse(accNet, accData)
end

# Load FMUs
fmus = Vector{FMU2}()
for i in 1:2 # how many FMUs do you want?
    _fmu = fmi2Load("SpringPendulumExtForce1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:CS)
    push!(fmus, _fmu)
end

# NeuralFMU setup
total_fmu_outdim = sum(map(x->length(x.modelDescription.outputValueReferences), fmus))

evalFMU = function(i, u)
    fmus[i](;u_refs=fmus[i].modelDescription.inputValueReferences, u=u, y_refs=fmus[i].modelDescription.outputValueReferences)
end
net = Chain(
    Parallel(
        vcat,
        inputs -> evalFMU(1, inputs[1:1]),
        inputs -> evalFMU(2, inputs[2:2])
    ),
    Dense(total_fmu_outdim, 16, tanh; init=Flux.identity_init),
    Dense(16, 16, tanh; init=Flux.identity_init),
    Dense(16, length(fmus[1].modelDescription.outputValueReferences); init=Flux.identity_init),
)

problem = CS_NeuralFMU(fmus, net, (t_start, t_stop))
@test problem != nothing

solutionBefore = problem(extForce, t_step)

# train it ...
p_net = Flux.params(problem)

optim = OPTIMISER(ETA)

lossBefore = losssum(p_net[1])
FMIFlux.train!(losssum, problem, Iterators.repeated((), NUMSTEPS), optim; gradient=GRADIENT)
lossAfter = losssum(p_net[1])
@test lossAfter < lossBefore

# check results
solutionAfter = problem(extForce, t_step)

for i in 1:length(fmus)
    fmi2Unload(fmus[i])
end
