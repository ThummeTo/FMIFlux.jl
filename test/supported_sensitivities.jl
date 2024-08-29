#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using DifferentialEquations

import Random 
Random.seed!(5678);

# boundaries
t_start = 0.0
t_step = 0.1
t_stop = 5.0 
tData = t_start:t_step:t_stop
tspan = (t_start, t_stop)
posData = ones(Float64, length(tData))

# load FMU for NeuralFMU
fmu = loadFMU("BouncingBall", "ModelicaReferenceFMUs", "0.0.25"; type=:ME)
fmu.handleEventIndicators = [UInt32(1)] 

x0_bb = [1.0, 0.0]
numStates = length(x0_bb)

net = Chain(x -> fmu(;x=x, dx_refs=:all), 
            Dense(2, 16, tanh),
            Dense(16, 2, identity))

# loss function for training
losssum = function(p)
    global nfmu, x0_bb, posData
    solution = nfmu(x0_bb; p=p, saveat=tData)

    if !solution.success
        return Inf 
    end

    posNet = getState(solution, 1; isIndex=true)
    
    return FMIFlux.Losses.mse(posNet, posData)
end

solvers = [Tsit5(), FBDF(autodiff=false)] # , FBDF(autodiff=true)]
for solver in solvers
    
    global nfmu
    @info "Solver: $(solver)"
    nfmu = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver; saveat=tData) 
    nfmu.modifiedState = false
    nfmu.snapshots = true

    best_timing, best_gradient, best_sensealg = FMIFlux.checkSensalgs!(losssum, nfmu)
    @test best_timing != Inf
end

unloadFMU(fmu)