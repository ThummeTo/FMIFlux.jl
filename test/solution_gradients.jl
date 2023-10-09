#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using DifferentialEquations
using FMIFlux, FMIZoo, Test
import FMISensitivity.SciMLSensitivity.SciMLBase: RightRootFind, LeftRootFind
import FMIFlux: unsense
using ForwardDiff, ReverseDiff, FiniteDiff, Zygote
EXPORTINGTOOL = "Dymola"
EXPORTINGVERSION = "2022x"

import Random 
Random.seed!(5678);

ENERGY_LOSS = 0.7 
RADIUS = 0.0
GRAVITY = 9.81
DBL_MIN = 2.2250738585072013830902327173324040642192159804623318306e-308

t_start = 0.0
t_step = 0.05
t_stop = 2.0
tData = t_start:t_step:t_stop
posData = ones(length(tData))

x0 = [1.0, 0.0]
dx = [0.0, 0.0]
numStates = 2
solver = Tsit5()

# setup BouncingBallODE 
function fx_bb(x)
    dx = [x[2], -GRAVITY]
    dx 
end
function fx(dx, x, p, t)
    # if rand(1:10)%10 == 0
    #     @info "$(typeof(x))"
    # end
    dx[:] = re_bb(p)(x)
end

net_bb = Chain(fx_bb,
            Dense([1.0 0.0; 0.0 1.0], [0.0, 0.0], identity))

ff = ODEFunction{true}(fx) 
prob_bb = ODEProblem{true}(ff, x0, (t_start, t_stop), ())

function condition(out, x, t, integrator)
    out[1] = x[1]-RADIUS
    #out[2] = x[1]-RADIUS
end

global events = 0
function affect_right!(integrator, idx)
    s_new = RADIUS + DBL_MIN
    v_new = -integrator.u[2]*ENERGY_LOSS
    u_new = [s_new, v_new]

    global events 
    events += 1
    @info "[$(events)] New state at $(integrator.t) is $(u_new)"

    integrator.u .= u_new
end
function affect_left!(integrator, idx)
    s_new = integrator.u[1]
    v_new = -integrator.u[2]*ENERGY_LOSS
    u_new = [s_new, v_new]

    global events 
    events += 1
    @info "[$(events)] New state at $(integrator.t) is $(u_new)"

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
#fmu = fmiLoad("BouncingBall1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:ME)
fmu = fmiLoad("BouncingBall", "ModelicaReferenceFMUs", "0.0.25")
fmu.handleEventIndicators = nothing

net = Chain(x -> fmu(;x=x, dx=dx), 
            Dense([1.0 0.0; 0.0 1.0], [0.0; 0.0], identity))

prob = ME_NeuralFMU(fmu, net, (t_start, t_stop); saveat=tData) 
prob.modifiedState = false

# ANNs
global solution = nothing 

function losssum(p; sensealg=nothing)
    global solution, prob, x0, posData, solver
    solution = prob(x0; p=p, solver=solver, sensealg=sensealg)

    if !solution.success
        @error "Solution failed!"
        return Inf 
    end

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    return Flux.Losses.mae(posNet, posData)
end

function losssum_bb(p; sensealg=nothing, root=:Right)
    
    posNet = mysolve(p; sensealg=sensealg, root=root)
    
    return Flux.Losses.mae(posNet, posData)
end

function mysolve(p; sensealg=nothing, root=:Right)
    global solution, prob_bb, x0, posData, solver, events 
    events = 0

    callback = nothing
    if root == :Right 
        callback = CallbackSet(rightCb)
    else
        callback = CallbackSet(leftCb)
    end
    solution = solve(prob_bb, solver; p=p, saveat=tData, callback=callback, sensealg=sensealg)

    if !isa(solution, AbstractArray)
        if solution.retcode != FMI.ReturnCode.Success
            @error "Solution failed!"
            return Inf 
        end

        return solution.u
    else
        return solution[1,:]
    end
end

p_net = Flux.params(prob)[1]
p_net_bb, re_bb = Flux.destructure(net_bb)

using SciMLSensitivity, Plots
sensealg = ReverseDiffAdjoint() # QuadratureAdjoint(autojacvec=ReverseDiffVJP())

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
jac_con1 = ForwardDiff.jacobian(condition_bb_check, x0)
jac_con2 = ForwardDiff.jacobian(condition_nfmu_check, x0)

jac_con1 = ReverseDiff.jacobian(condition_bb_check, x0)
jac_con2 = ReverseDiff.jacobian(condition_nfmu_check, x0)

### START CHECK AFFECT

import SciMLSensitivity: u_modified!
import FMI: fmi2SimulateME
function u_modified!(::NamedTuple, ::Any)
    return nothing
end
function affect_bb_check(x)
    integrator = (t=t_start, u=x)
    affect_right!(integrator, 1)
    return integrator.u
end
function affect_nfmu_check(x)
    integrator = (t=t_start, u=x, opts=(internalnorm=(a,b)->1.0,) )
    #FMIFlux.affectFMU!(prob, FMIFlux.getComponent(prob), integrator, 1)
    integrator.u[1] = DBL_MIN
    integrator.u[2] = -0.7 * integrator.u[2]
    return integrator.u
end
t_first_event_time = 0.451523640985728
x_first_event_right = [2.2250738585072014e-308, 3.1006128426489954]
#sol = fmi2SimulateME(fmu, (t_start, t_first_event_time); solver=solver)
jac_con1 = ForwardDiff.jacobian(affect_bb_check, x0)
jac_con2 = ForwardDiff.jacobian(affect_nfmu_check, x0)

jac_con1 = ReverseDiff.jacobian(affect_bb_check, x0)
jac_con2 = ReverseDiff.jacobian(affect_nfmu_check, x0)

###

fig = plot(;ylims=(-0.1, 1.1)) 

# Solution (plain)
losssum(p_net; sensealg=sensealg) 
length(solution.events)
plot!(fig, tData, collect(u[1] for u in solution.states.u); label="NFMU: Sol")

losssum_bb(p_net_bb; sensealg=sensealg) 
events
plot!(fig, tData, collect(u[1] for u in solution.u); label="NODE: Sol")

# Solution FWD
grad_fwd1 = ForwardDiff.gradient(p -> losssum(p; sensealg=sensealg), p_net)
length(solution.events)
plot!(fig, tData, collect(unsense(u[1]) for u in solution.states.u); label="NFMU: FWD")

grad_fwd2 = ForwardDiff.gradient(p -> losssum_bb(p; sensealg=sensealg), p_net_bb)
events
plot!(fig, tData, collect(unsense(u[1]) for u in solution.u); label="NODE: FWD")

# Solution ReverseDiff
grad_rwd1 = ReverseDiff.gradient(p -> losssum(p; sensealg=sensealg), p_net)
length(solution.events)
plot!(fig, tData, collect(unsense(u[1]) for u in solution.states.u); label="NFMU: RWD")

grad_rwd2 = ReverseDiff.gradient(p -> losssum_bb(p; sensealg=sensealg), p_net_bb)
events
plot!(fig, tData, collect(unsense(u[1]) for u in solution[1,:]); label="NODE: RWD")

# Solution Zygote
# grad_zyg1 = Zygote.gradient(p -> losssum(p; sensealg=sensealg), p_net)[1]
# plot!(fig, tData, collect(unsense(u[1]) for u in solution.states.u); label="NFMU: ZYG")

# grad_zyg2 = Zygote.gradient(p -> losssum_bb(p; sensealg=sensealg), p_net_bb)[1]
# plot!(fig, tData, collect(unsense(u[1]) for u in solution[1,:]); label="NODE: ZYG")

# Ground Truth
grad_fin1 = FiniteDiff.finite_difference_gradient(p -> losssum_bb(p; sensealg=sensealg), p_net_bb, Val{:central}; absstep=1e-8)
grad_fin2 = FiniteDiff.finite_difference_gradient(p -> losssum(p; sensealg=sensealg), p_net, Val{:central}; absstep=1e-8)

atol = 1e-5
@test isapprox(grad_fin1, grad_fwd1; atol=atol)
@test isapprox(grad_fin2, grad_fwd2; atol=atol)

@test isapprox(grad_fin1, grad_rwd1; atol=atol)
@test isapprox(grad_fin2, grad_rwd2; atol=atol)

# Jacobina Test

jac_fwd1 = ForwardDiff.jacobian(p -> mysolve(p; sensealg=sensealg), p_net)
plot(tData, collect(unsense(u[1]) for u in solution.u); label="FWD")
jac_rwd1 = ReverseDiff.jacobian(p -> mysolve(p; sensealg=sensealg), p_net)
plot!(tData, collect(unsense(u[1]) for u in solution[1,:]); label="RWD (Right)")
jac_rwd1 = ReverseDiff.jacobian(p -> mysolve(p; sensealg=sensealg, root=:Left), p_net)
plot!(tData, collect(unsense(u[1]) for u in solution[1,:]); label="RWD (Left)")
jac_fin1 = FiniteDiff.finite_difference_jacobian(p -> jac_bb(p; sensealg=sensealg), p_net)

###

atol = 1e-4
@test isapprox(grad_fin1, grad_fwd1; atol=atol)
@test isapprox(grad_fin2, grad_fwd2; atol=atol)

@test isapprox(grad_fin1, grad_rwd1; atol=atol)
@test isapprox(grad_fin2, grad_rwd2; atol=atol)

# atol = 1e-4
# @test isapprox(collect(u[1] for u in sol.states.u), collect(u[1] for u in sol_fin.states.u); atol=atol)
# @test isapprox(collect(u[1] for u in sol.states.u), collect(u[1] for u in sol_fwd.states.u); atol=atol)
# @test isapprox(collect(u[1] for u in sol.states.u), collect(u[1] for u in sol_rwd.states.u); atol=atol)

###

fmiUnload(fmu)
