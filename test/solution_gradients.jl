#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using DifferentialEquations
using FMIFlux, FMIZoo, Test
import FMIFlux.FMISensitivity.SciMLSensitivity.SciMLBase: RightRootFind, LeftRootFind
import FMIFlux: unsense
using FMIFlux.FMISensitivity.SciMLSensitivity.ForwardDiff, FMIFlux.FMISensitivity.SciMLSensitivity.ReverseDiff, FMIFlux.FMISensitivity.SciMLSensitivity.FiniteDiff, FMIFlux.FMISensitivity.SciMLSensitivity.Zygote

import Random 
Random.seed!(5678);

global solution = nothing
global events = 0

ENERGY_LOSS = 0.7 
RADIUS = 0.0
GRAVITY = 9.81
DBL_MIN = 2.2250738585072013830902327173324040642192159804623318306e-308

NUMEVENTS = 4

t_start = 0.0
t_step = 0.05
t_stop = 2.0
tData = t_start:t_step:t_stop
posData = ones(Float64, length(tData))
x0_bb = [1.0, 0.0]

numStates = 2
solver = Tsit5()

# setup BouncingBallODE 
function fx(x)
    return [x[2], -GRAVITY] 
end

function fx_bb(dx, x, p, t)
    dx[:] = re_bb(p)(x)
    return nothing
end

net_bb = Chain(#Dense([1.0 0.0; 0.0 1.0], [0.0, 0.0], identity),
            fx,
            Dense([1.0 0.0; 0.0 1.0], [0.0, 0.0], identity))
p_net_bb, re_bb = Flux.destructure(net_bb)

ff = ODEFunction{true}(fx_bb) 
prob_bb = ODEProblem{true}(ff, x0_bb, (t_start, t_stop), p_net_bb)

function condition(out, x, t, integrator)
    out[1] = x[1]-RADIUS
    #out[2] = x[1]-RADIUS 
end

function affect_right!(integrator, idx)
    s_new = RADIUS + DBL_MIN
    v_new = -integrator.u[2]*ENERGY_LOSS
    u_new = [s_new, v_new]

    global events 
    events += 1
    # @info "[$(events)] New state at $(integrator.t) is $(u_new)"

    integrator.u .= u_new
end
function affect_left!(integrator, idx)
    s_new = integrator.u[1]
    v_new = -integrator.u[2]*ENERGY_LOSS
    u_new = [s_new, v_new]

    global events 
    events += 1
    # @info "[$(events)] New state at $(integrator.t) is $(u_new)"

    integrator.u .= u_new
end

rightCb = VectorContinuousCallback(condition,
                                   affect_right!,
                                   1;
                                   rootfind=RightRootFind, save_positions=(false, false))
leftCb = VectorContinuousCallback(condition,
                                   affect_left!,
                                   1;
                                   rootfind=LeftRootFind, save_positions=(false, false))

# load FMU for NeuralFMU
fmu = fmi2Load("BouncingBall", "ModelicaReferenceFMUs", "0.0.25"; type=:ME)
fmu.handleEventIndicators = nothing

net = Chain(#Dense([1.0 0.0; 0.0 1.0], [0.0, 0.0], identity),
            x -> fmu(;x=x, dx_refs=:all), 
            Dense([1.0 0.0; 0.0 1.0], [0.0; 0.0], identity))

prob = ME_NeuralFMU(fmu, net, (t_start, t_stop)) 
prob.modifiedState = false

# ANNs 

function losssum(p; sensealg=nothing)
    global posData
    posNet = mysolve(p; sensealg=sensealg)

    return Flux.Losses.mae(posNet, posData)
end

function losssum_bb(p; sensealg=nothing, root=:Right)
    global posData
    posNet = mysolve_bb(p; sensealg=sensealg, root=root)
    
    return Flux.Losses.mae(posNet, posData)
end

function mysolve(p; sensealg=nothing)
    global solution, events # write
    global prob, x0_bb, posData, solver # read-only
    events = 0

    solution = prob(x0_bb; p=p, solver=solver, saveat=tData)

    return collect(u[1] for u in solution.states.u)
end

function mysolve_bb(p; sensealg=nothing, root=:Right)
    global solution # write 
    global prob_bb, solver, events # read
    events = 0

    callback = nothing
    if root == :Right 
        callback = CallbackSet(rightCb)
    else
        callback = CallbackSet(leftCb)
    end
    solution = solve(prob_bb; p=p, alg=solver, saveat=tData, callback=callback, sensealg=sensealg)

    if !isa(solution, AbstractArray)
        if solution.retcode != FMI.ReturnCode.Success
            @error "Solution failed!"
            return Inf 
        end

        return collect(u[1] for u in solution.u)
    else
        return solution[1,:] # collect(solution[:,i] for i in 1:size(solution)[2]) 
    end
end

p_net = Flux.params(prob)[1]

using FMIFlux.FMISensitivity.SciMLSensitivity
sensealg = ReverseDiffAdjoint() 

c = nothing
c, _ = FMIFlux.prepareSolveFMU(prob.fmu, c, fmi2TypeModelExchange, nothing, nothing, nothing, nothing, nothing, prob.parameters, prob.tspan[1], prob.tspan[end], nothing; x0=prob.x0, handleEvents=FMIFlux.handleEvents, cleanup=true)

### START CHECK CONDITIONS 

function condition_bb_check(x)
    buffer = similar(x, 1)
    condition(buffer, x, t_start, nothing)
    return buffer 
end
function condition_nfmu_check(x)
    buffer = similar(x, 1)
    FMIFlux.condition!(prob, FMIFlux.getComponent(prob), buffer, x, t_start, nothing, [UInt32(1)])
    return buffer 
end
jac_fwd1 = ForwardDiff.jacobian(condition_bb_check, x0_bb)
jac_fwd2 = ForwardDiff.jacobian(condition_nfmu_check, x0_bb)

jac_rwd1 = ReverseDiff.jacobian(condition_bb_check, x0_bb)
jac_rwd2 = ReverseDiff.jacobian(condition_nfmu_check, x0_bb)

jac_fin1 = FiniteDiff.finite_difference_jacobian(condition_bb_check, x0_bb)
jac_fin2 = FiniteDiff.finite_difference_jacobian(condition_nfmu_check, x0_bb)

atol = 1e-8
@test isapprox(jac_fin1, jac_fwd1; atol=atol)
@test isapprox(jac_fin1, jac_rwd1; atol=atol)
@test isapprox(jac_fin2, jac_fwd2; atol=atol)
@test isapprox(jac_fin2, jac_rwd2; atol=atol)

### START CHECK AFFECT

# import SciMLSensitivity: u_modified!
# import FMI: fmi2SimulateME
# function u_modified!(::NamedTuple, ::Any)
#     return nothing
# end
# function affect_bb_check(x)
#     integrator = (t=t_start, u=x)
#     affect_right!(integrator, 1)
#     return integrator.u
# end
# function affect_nfmu_check(x)
#     integrator = (t=t_start, u=x, opts=(internalnorm=(a,b)->1.0,) )
#     #FMIFlux.affectFMU!(prob, FMIFlux.getComponent(prob), integrator, 1)
#     integrator.u[1] = DBL_MIN
#     integrator.u[2] = -0.7 * integrator.u[2]
#     return integrator.u
# end
# t_first_event_time = 0.451523640985728
# x_first_event_right = [2.2250738585072014e-308, 3.1006128426489954]

# jac_con1 = ForwardDiff.jacobian(affect_bb_check, x0_bb)
# jac_con2 = ForwardDiff.jacobian(affect_nfmu_check, x0_bb)

# jac_con1 = ReverseDiff.jacobian(affect_bb_check, x0_bb)
# jac_con2 = ReverseDiff.jacobian(affect_nfmu_check, x0_bb)

###

# Solution (plain)
losssum(p_net; sensealg=sensealg) 
@test length(solution.events) == NUMEVENTS

losssum_bb(p_net_bb; sensealg=sensealg) 
@test events == NUMEVENTS

# Solution FWD
grad_fwd1 = ForwardDiff.gradient(p -> losssum(p; sensealg=sensealg), p_net)
#@test length(solution.events) == NUMEVENTS

grad_fwd2 = ForwardDiff.gradient(p -> losssum_bb(p; sensealg=sensealg), p_net_bb)
@test events == NUMEVENTS

# Solution ReverseDiff
grad_rwd1 = ReverseDiff.gradient(p -> losssum(p; sensealg=sensealg), p_net)
#@test length(solution.events) == NUMEVENTS

grad_rwd2 = ReverseDiff.gradient(p -> losssum_bb(p; sensealg=sensealg), p_net_bb)
@test events == NUMEVENTS

# Ground Truth
grad_fin1 = FiniteDiff.finite_difference_gradient(p -> losssum(p; sensealg=sensealg), p_net, Val{:central}; absstep=1e-8)
grad_fin2 = FiniteDiff.finite_difference_gradient(p -> losssum_bb(p; sensealg=sensealg), p_net_bb, Val{:central}; absstep=1e-8)

atol = 1e-5
@test isapprox(grad_fin1, grad_fwd1; atol=atol)
@test isapprox(grad_fin2, grad_fwd2; atol=atol)

@test isapprox(grad_fin1, grad_rwd1; atol=atol)
@test isapprox(grad_fin2, grad_rwd2; atol=atol)

# Jacobian Test

jac_fwd1 = ForwardDiff.jacobian(p -> mysolve_bb(p; sensealg=sensealg), p_net)
jac_fwd2 = ForwardDiff.jacobian(p -> mysolve(p; sensealg=sensealg), p_net)

jac_rwd1 = ReverseDiff.jacobian(p -> mysolve_bb(p; sensealg=sensealg), p_net)
jac_rwd2 = ReverseDiff.jacobian(p -> mysolve(p; sensealg=sensealg), p_net)

# [TODO] why this?!
jac_rwd1[2:end,:] = jac_rwd1[2:end,:] .- jac_rwd1[1:end-1,:]
jac_rwd2[2:end,:] = jac_rwd2[2:end,:] .- jac_rwd2[1:end-1,:]

jac_fin1 = FiniteDiff.finite_difference_jacobian(p -> mysolve_bb(p; sensealg=sensealg), p_net)
jac_fin2 = FiniteDiff.finite_difference_jacobian(p -> mysolve(p; sensealg=sensealg), p_net)

###

atol = 1e-4
@test isapprox(jac_fin1, jac_fwd1; atol=atol)
@test isapprox(jac_fin1, jac_rwd1; atol=atol)

@test isapprox(jac_fin2, jac_fwd2; atol=atol)
@test isapprox(jac_fin2, jac_rwd2; atol=atol)

###

fmi2Unload(fmu)
