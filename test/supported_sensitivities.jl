#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux

import Random 
Random.seed!(5678);

# setup BouncingBallODE 
function fx_bb(x)
    dx = [x[2], -9.81]
    dx 
end
function fx(dx, x, p, t)
    dx[:] = re(p)(x)
end

net = Chain(fx_bb,
            Dense([1.0 0.0; 0.0 1.0], [0.0, 0.0], identity))

ff = ODEFunction{true}(fx) # , tgrad=nothing)
bb_prob = ODEProblem{true}(ff, x0, tspan, params)

function condition(out, x, t, integrator)
    out[1] = x[1]
end

function affect!(integrator, idx)
    u_new = [max(integrator.u[1], 0.0), -integrator.u[2]*0.9]
    integrator.u .= u_new
end

eventCb = VectorContinuousCallback(condition,
                                   affect!,
                                   2;
                                   rootfind=RightRootFind, save_positions=(false, false))

t_start = 0.0
t_step = 0.1
t_stop = 5.0
tData = t_start:t_step:t_stop
posData = ones(length(tData))

# load FMU for NeuralFMU
fmu = fmiLoad("BouncingBall1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:ME)
fmu.handleEventIndicators = [1]

x0 = [1.0, 0.0]
dx = [0.0, 0.0]
numStates = length(x0)

net = Chain(x -> fmu(;x=x, dx=dx), 
            Dense([1.0 0.0; 0.0 1.0], [0.0; 0.0], identity))

# loss function for training
function losssum(p)
    global nfmu, x0, posData
    solution = nfmu(x0; p=p)

    if !solution.success
        return Inf 
    end

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    return FMIFlux.Losses.mse(posNet, posData)
end

function losssum_bb(p)
    global nfmu, x0, posData
    solution = nfmu(x0; p=p)

    if !solution.success
        return Inf 
    end

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    return FMIFlux.Losses.mse(posNet, posData)
end

nfmu = ME_NeuralFMU(fmu, net, (t_start, t_stop); saveat=tData) 
nfmu.modifiedState = false

using SciMLSensitivity
params = Flux.params(nfmu)
fmu.executionConfig.sensealg = ReverseDiffAdjoint() # QuadratureAdjoint(autojacvec=ReverseDiffVJP(false))
grad_fd = ForwardDiff.gradient(losssum, params[1])
grad_rd = ReverseDiff.gradient(losssum, params[1])
abc = 1

import ReverseDiff: increment_deriv!, ZeroTangent
function ReverseDiff.increment_deriv!(::ReverseDiff.TrackedReal, ::ZeroTangent)
    return nothing 
end

import DifferentialEquations.DiffEqBase: AbstractODEIntegrator
function Base.show(io::IO, ::MIME"text/plain", ::AbstractODEIntegrator)
    print(io, "[AbstractODEIntegrator]")
end

FMIFlux.checkSensalgs!(losssum, nfmu)

fmiUnload(fmu)