#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using FMI.DifferentialEquations

import Random 
Random.seed!(5678);

# boundaries
t_start = 0.0
t_step = 0.1
t_stop = 3.0
tData = t_start:t_step:t_stop
posData = ones(length(tData))
tspan = (t_start, t_stop)

# load FMU for NeuralFMU
fmu = fmiLoad("SpringFrictionPendulum1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:ME)

x0 = [1.0, 0.0]
numStates = length(x0)

c1 = CacheLayer()
c2 = CacheRetrieveLayer(c1)
c3 = CacheLayer()
c4 = CacheRetrieveLayer(c3)

# default ME-NeuralFMU (learn dynamics and states, almost-neutral setup, parameter count << 100)
net = Chain(x -> c1(x),
            Dense(numStates, 32, identity; init=Flux.identity_init),
            Dense(32, numStates, identity; init=Flux.identity_init),
            x -> c2([1], x[2], []),
            x -> fmu(;x=x), 
            x -> c3(x),
            Dense(numStates, 32, identity; init=Flux.identity_init),
            Dense(32, numStates, identity; init=Flux.identity_init),
            x -> c4([1], x[2], []))

# loss function for training
function losssum(p)
    global nfmu, x0, posData
    solution = nfmu(x0; p=p, saveat=tData)

    if !solution.success
        return Inf 
    end

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    return FMIFlux.Losses.mse(posNet, posData)
end

solver = Tsit5()
nfmu = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver; saveat=tData) 
nfmu.modifiedState = false

FMIFlux.checkSensalgs!(losssum, nfmu)

fmiUnload(fmu)