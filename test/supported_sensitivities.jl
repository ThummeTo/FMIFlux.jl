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
nfmu = nothing

# load FMU for NeuralFMU
fmus = []

# this is a non-simultaneous event system (one event per time instant)
f = loadFMU("BouncingBall", "ModelicaReferenceFMUs", "0.0.25"; type=:ME)
@assert f.modelDescription.numberOfEventIndicators == 1 "Wrong number of event indicators: $(f.modelDescription.numberOfEventIndicators) != 1"
push!(fmus, f) 

# this is a simultaneous event system (two events per time instant) 
f = loadFMU("BouncingBall1D", "Dymola", "2023x"; type=:ME)          
@assert f.modelDescription.numberOfEventIndicators == 2 "Wrong number of event indicators: $(f.modelDescription.numberOfEventIndicators) != 2"
push!(fmus, f) 

x0_bb = [1.0, 0.0]
numStates = length(x0_bb)

function net_const(fmu)
    net = Chain(x -> fmu(;x=x, dx_refs=:all), 
                Dense(2, 16, tanh),
                Dense(16, 2, identity))
    return net
end

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

for fmu in fmus
    @info "##### CHECKING FMU WITH $(fmu.modelDescription.numberOfEventIndicators) SIMULTANEOUS EVENT INDICATOR(S) #####"
    solvers = [Tsit5(), Rosenbrock23(autodiff=false), Rodas5(autodiff=false), FBDF(autodiff=false)] 
    for solver in solvers
        
        global nfmu
        @info "##### SOLVER: $(solver) #####"

        net = net_const(fmu)
        nfmu = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver; saveat=tData) 
        nfmu.modifiedState = false
        nfmu.snapshots = true

        best_timing, best_gradient, best_sensealg = FMIFlux.checkSensalgs!(losssum, nfmu)
        #@test best_timing != Inf
    end
end

unloadFMU(fmu)