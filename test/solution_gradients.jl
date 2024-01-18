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
using FMIFlux.FMIImport, FMIFlux.FMIImport.FMICore, FMIZoo
using FMIFlux.FMIImport.FMICore: unsense
import LinearAlgebra:I

import Random 
Random.seed!(5678);

global solution = nothing
global events = 0

ENERGY_LOSS = 0.7 
RADIUS = 0.0
GRAVITY = 9.81
DBL_MIN = 1e-10 # 2.2250738585072013830902327173324040642192159804623318306e-308

NUMEVENTS = 4

t_start = 0.0
t_step = 0.05
t_stop = 2.0
tData = t_start:t_step:t_stop
posData = ones(Float64, length(tData))
x0_bb = [1.0, 0.0]

numStates = 2
solver = Tsit5()

Wr = zeros(2,2) # rand(2,2)*1e-12
br = zeros(2) #rand(2)*1e-12

W1 = [1.0 0.0; 0.0 1.0]         - Wr
b1 = [0.0, 0.0]                 - br
W2 = [1.0 0.0; 0.0 1.0]         - Wr
b2 = [0.0, 0.0]                 - br

∂xn_∂xp = [0.0 0.0; 0.0 -ENERGY_LOSS]

# setup BouncingBallODE 
fx = function(x)
    return [x[2], -GRAVITY] 
end

fx_bb = function(dx, x, p, t)
    dx[:] = re_bb(p)(x)
    return nothing
end

net_bb = Chain(#Dense(W1, b1, identity),
            fx,
            Dense(W2, b2, identity))
p_net_bb, re_bb = Flux.destructure(net_bb)

ff = ODEFunction{true}(fx_bb) 
prob_bb = ODEProblem{true}(ff, x0_bb, (t_start, t_stop), p_net_bb)

condition = function(out, x, t, integrator)
    #x = re_bb(p_net_bb)[1](x)
    out[1] = x[1]-RADIUS
end

time_choice = function(integrator)
    ts = [0.451523640985728, 1.083656738365748, 1.5261499065317576, 1.8358951242479626]
    i = 1 
    while ts[i] <= integrator.t 
        i += 1

        if i > length(ts)
            return nothing 
        end
    end

    return ts[i]
end

affect_right! = function(integrator, idx)

    #@info "affect_right! triggered by #$(idx)"

    # if idx == 1
    #     # event #1 is handeled as "dummy" (e.g. discrete state change)
    #     return 
    # end

    if idx > 0
        out = zeros(1)
        x = integrator.u
        t = integrator.t
        condition(out, unsense(x), unsense(t), integrator)
        if sign(out[idx]) > 0.0
            @info "Event for bouncing ball (white-box) triggered, but not valid!"
            return nothing 
        end
    end
    
    s_new = RADIUS + DBL_MIN
    v_new = -integrator.u[2]*ENERGY_LOSS
    u_new = [s_new, v_new]

    global events 
    events += 1
    #@info "[$(events)] New state at $(integrator.t) is $(u_new) triggered by #$(idx)"

    integrator.u .= u_new
end
affect_left! = function(integrator, idx)

    #@info "affect_left! triggered by #$(idx)"

    # if idx == 1
    #     # event #1 is handeled as "dummy" (e.g. discrete state change)
    #     return 
    # end

    out = zeros(1)
    x = integrator.u
    t = integrator.t
    condition(out, unsense(x), unsense(t), integrator)
    if sign(out[idx]) < 0.0
        @warn "Event for bouncing ball triggered, but not valid!"
        return nothing 
    end

    s_new = integrator.u[1]
    v_new = -integrator.u[2]*ENERGY_LOSS
    u_new = [s_new, v_new]

    global events 
    events += 1
    #@info "[$(events)] New state at $(integrator.t) is $(u_new)"

    integrator.u .= u_new
end

NUMEVENTINDICATORS = 1 # 2
rightCb = VectorContinuousCallback(condition, #_double,
                                   affect_right!,
                                   NUMEVENTINDICATORS;
                                   rootfind=RightRootFind, save_positions=(false, false))
leftCb = VectorContinuousCallback(condition, #_double,
                                   affect_left!,
                                   NUMEVENTINDICATORS;
                                   rootfind=LeftRootFind, save_positions=(false, false))

timeCb = IterativeCallback(time_choice,
                    (indicator) -> affect_right!(indicator, 0), 
                    Float64; 
                    initial_affect=false,
                    save_positions=(false, false))

# load FMU for NeuralFMU
#fmu = fmi2Load("BouncingBall", "ModelicaReferenceFMUs", "0.0.25"; type=:ME)
#fmu_params = nothing
fmu = fmi2Load("BouncingBall1D", "Dymola", "2022x"; type=:ME)
fmu_params = Dict("damping" => 0.7, "mass_radius" => 0.0, "mass_s_min" => DBL_MIN)
fmu.executionConfig.isolatedStateDependency = true

net = Chain(#Dense(W1, b1, identity),
            x -> fmu(;x=x, dx_refs=:all), 
            Dense(W2, b2, identity))

prob = ME_NeuralFMU(fmu, net, (t_start, t_stop)) 
prob.snapshots = true # needed for correct snesitivities

# ANNs 

losssum = function(p; sensealg=nothing)
    global posData
    posNet = mysolve(p; sensealg=sensealg)

    return Flux.Losses.mae(posNet, posData)
end

losssum_bb = function(p; sensealg=nothing, root=:Right)
    global posData
    posNet = mysolve_bb(p; sensealg=sensealg, root=root)
    
    return Flux.Losses.mae(posNet, posData)
end

mysolve = function(p; sensealg=nothing)
    global solution, events # write
    global prob, x0_bb, posData, solver # read-only
    events = 0

    solution = prob(x0_bb; p=p, solver=solver, saveat=tData, parameters=fmu_params)

    return collect(u[1] for u in solution.states.u)
end

mysolve_bb = function(p; sensealg=nothing, root=:Right)
    global solution # write 
    global prob_bb, solver, events # read
    events = 0

    callback = nothing
    if root == :Right 
        callback = CallbackSet(rightCb)
    elseif root == :Left
        callback = CallbackSet(leftCb)
    elseif root == :Time
        callback = CallbackSet(timeCb)
    else
        @assert false "unknwon root `$(root)`"
    end
    solution = solve(prob_bb; p=p, alg=solver, saveat=tData, callback=callback, sensealg=sensealg)

    if !isa(solution, AbstractArray)
        if solution.retcode != FMIFlux.ReturnCode.Success
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
sensealg = ReverseDiffAdjoint() # InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)) #  

c = nothing
c, _ = FMIFlux.prepareSolveFMU(prob.fmu, c, fmi2TypeModelExchange, nothing, nothing, nothing, nothing, nothing, prob.parameters, prob.tspan[1], prob.tspan[end], nothing; x0=prob.x0, handleEvents=FMIFlux.handleEvents, cleanup=true)

### START CHECK CONDITIONS 

condition_bb_check = function(x)
    buffer = similar(x, 1)
    condition(buffer, x, t_start, nothing)
    return buffer 
end
condition_nfmu_check = function(x)
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

atol = 1e-6
@test isapprox(jac_fin1, jac_fwd1; atol=atol)
@test isapprox(jac_fin1, jac_rwd1; atol=atol)
@test isapprox(jac_fin2, jac_fwd2; atol=atol)
@test isapprox(jac_fin2, jac_rwd2; atol=atol)

### START CHECK AFFECT

affect_bb_check = function(x)

    # convert TrackedArrays to Array{<:TrackedReal,1}
    if !isa(x, AbstractVector{<:Float64})
        x = [x...]
    else
        x = copy(x)
    end

    integrator = (t=t_start, u=x)
    affect_right!(integrator, 1)
    return integrator.u
end
affect_nfmu_check = function(x)
    global prob

    # convert TrackedArrays to Array{<:TrackedReal,1}
    if !isa(x, AbstractVector{<:Float64})
        x = [x...]
    else
        x = copy(x)
    end
    
    c, _ = FMIFlux.prepareSolveFMU(prob.fmu, nothing, fmi2TypeModelExchange, nothing, nothing, nothing, nothing, nothing, fmu_params, prob.tspan[1], prob.tspan[end], nothing; x0=unsense(x), handleEvents=FMIFlux.handleEvents, cleanup=true)

    integrator = (t=t_start, u=x, opts=(internalnorm=(a,b)->1.0,) )
    FMIFlux.affectFMU!(prob, c, integrator, 1)
    
    return integrator.u
end
#t_event_time = 0.451523640985728
x_event_left = [-1.0, -1.0] # [-3.808199081191736e-15, -4.429446918069994]
x_event_right = [0.0, 0.7] # [2.2250738585072014e-308, 3.1006128426489954]
x_no_event = [0.1, -1.0]

@test isapprox(affect_bb_check(x_event_left), x_event_right; atol=1e-4)
@test isapprox(affect_nfmu_check(x_event_left), x_event_right; atol=1e-4)

jac_con1 = ForwardDiff.jacobian(affect_bb_check, x_event_left)
jac_con2 = ForwardDiff.jacobian(affect_nfmu_check, x_event_left)

@test isapprox(jac_con1, ∂xn_∂xp; atol=1e-4)
@test isapprox(jac_con2, ∂xn_∂xp; atol=1e-4)

jac_con1 = ReverseDiff.jacobian(affect_bb_check, x_event_left)
jac_con2 = ReverseDiff.jacobian(affect_nfmu_check, x_event_left)

@test isapprox(jac_con1, ∂xn_∂xp; atol=1e-4)
@test isapprox(jac_con2, ∂xn_∂xp; atol=1e-4)

# [Note] checking via FiniteDiff is not possible here, because finite differences offsets might not trigger the events at all

# no-event 

@test isapprox(affect_bb_check(x_no_event), x_no_event; atol=1e-4)
@test isapprox(affect_nfmu_check(x_no_event), x_no_event; atol=1e-4)

jac_con1 = ForwardDiff.jacobian(affect_bb_check, x_no_event)
jac_con2 = ForwardDiff.jacobian(affect_nfmu_check, x_no_event)

@test isapprox(jac_con1, I; atol=1e-4)
@test isapprox(jac_con2, I; atol=1e-4)

jac_con1 = ReverseDiff.jacobian(affect_bb_check, x_no_event)
jac_con2 = ReverseDiff.jacobian(affect_nfmu_check, x_no_event)

@test isapprox(jac_con1, I; atol=1e-4)
@test isapprox(jac_con2, I; atol=1e-4)

###

# Solution (plain)
losssum(p_net; sensealg=sensealg) 
@test length(solution.events) == NUMEVENTS

losssum_bb(p_net_bb; sensealg=sensealg) 
@test events == NUMEVENTS

# Solution FWD (FMU)
grad_fwd_f = ForwardDiff.gradient(p -> losssum(p; sensealg=sensealg), p_net)
@test length(solution.events) == NUMEVENTS 

# Solution FWD (right)
root = :Right
grad_fwd_r = ForwardDiff.gradient(p -> losssum_bb(p; sensealg=sensealg, root=root), p_net_bb)
@test events == NUMEVENTS

# Solution FWD (left)
root = :Left
grad_fwd_l = ForwardDiff.gradient(p -> losssum_bb(p; sensealg=sensealg, root=root), p_net_bb)
@test events == NUMEVENTS

# Solution FWD (time)
root = :Time
grad_fwd_t = ForwardDiff.gradient(p -> losssum_bb(p; sensealg=sensealg, root=root), p_net_bb)
@test events == NUMEVENTS

# Solution RWD (FMU)
grad_rwd_f = ReverseDiff.gradient(p -> losssum(p; sensealg=sensealg), p_net)
@test length(solution.events) == NUMEVENTS 

# Solution RWD (right)
root = :Right
grad_rwd_r = ReverseDiff.gradient(p -> losssum_bb(p; sensealg=sensealg, root=root), p_net_bb)
@test events == NUMEVENTS

# Solution RWD (left)
root = :Left
grad_rwd_l = ReverseDiff.gradient(p -> losssum_bb(p; sensealg=sensealg, root=root), p_net_bb)
@test events == NUMEVENTS

# Solution RWD (time)
root = :Time
grad_rwd_t = ReverseDiff.gradient(p -> losssum_bb(p; sensealg=sensealg, root=root), p_net_bb)
@test events == NUMEVENTS

# Ground Truth
grad_fin_r = FiniteDiff.finite_difference_gradient(p -> losssum_bb(p; sensealg=sensealg, root=:Right), p_net_bb, Val{:central}; absstep=1e-6)
grad_fin_l = FiniteDiff.finite_difference_gradient(p -> losssum_bb(p; sensealg=sensealg, root=:Left), p_net_bb, Val{:central}; absstep=1e-6)
grad_fin_t = FiniteDiff.finite_difference_gradient(p -> losssum_bb(p; sensealg=sensealg, root=:Time), p_net_bb, Val{:central}; absstep=1e-6)
grad_fin_f = FiniteDiff.finite_difference_gradient(p -> losssum(p; sensealg=sensealg), p_net, Val{:central}; absstep=1e-6)

atol = 1e-3
inds = collect(1:length(p_net))
#deleteat!(inds, 1:6)

# check if finite differences match together
@test isapprox(grad_fin_f[inds], grad_fin_r[inds]; atol=atol)
@test isapprox(grad_fin_f[inds], grad_fin_l[inds]; atol=atol)

@test isapprox(grad_fin_f[inds], grad_fwd_f[inds]; atol=atol)
@test isapprox(grad_fin_f[inds], grad_rwd_f[inds]; atol=atol)

# Jacobian Test

jac_fwd_r = ForwardDiff.jacobian(p -> mysolve_bb(p; sensealg=sensealg), p_net)
jac_fwd_f = ForwardDiff.jacobian(p -> mysolve(p; sensealg=sensealg), p_net)

jac_rwd_r = ReverseDiff.jacobian(p -> mysolve_bb(p; sensealg=sensealg), p_net)
jac_rwd_f = ReverseDiff.jacobian(p -> mysolve(p; sensealg=sensealg), p_net)

# [TODO] why this?!
jac_rwd_r[2:end,:] = jac_rwd_r[2:end,:] .- jac_rwd_r[1:end-1,:]
jac_rwd_f[2:end,:] = jac_rwd_f[2:end,:] .- jac_rwd_f[1:end-1,:]

jac_fin_r = FiniteDiff.finite_difference_jacobian(p -> mysolve_bb(p; sensealg=sensealg), p_net)
jac_fin_f = FiniteDiff.finite_difference_jacobian(p -> mysolve(p; sensealg=sensealg), p_net)

###

atol = 1e-2
@test isapprox(jac_fin_f[:, inds], jac_fin_r[:, inds]; atol=atol)
@test isapprox(jac_fin_f[:, inds], jac_fwd_f[:, inds]; atol=atol)
@test isapprox(jac_fin_f[:, inds], jac_rwd_f[:, inds]; atol=atol)

###

fmi2Unload(fmu)
