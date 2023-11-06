#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using FMIFlux.DifferentialEquations

import Random 
Random.seed!(5678);

# boundaries
t_start = 0.0
t_step = 0.1
t_stop = 5.0
tData = t_start:t_step:t_stop
posData = ones(length(tData))
tspan = (t_start, t_stop)

# load FMU for NeuralFMU
fmu = fmi2Load("BouncingBall1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:ME)
fmu.handleEventIndicators = [1]

x0 = [1.0, 0.0]
dx = [0.0, 0.0]
numStates = length(x0)

net = Chain(x -> fmu(;x=x, dx=dx), 
            Dense([1.0 0.0; 0.0 1.0], [0.0; 0.0], identity))

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

fmi2Unload(fmu)