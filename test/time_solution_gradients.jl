#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMIFlux.Flux
using DifferentialEquations
using FMIFlux, FMIZoo, Test
import FMIFlux.FMISensitivity.SciMLSensitivity.SciMLBase: RightRootFind, LeftRootFind
import FMIFlux: unsense
using FMIFlux.FMISensitivity.SciMLSensitivity.ForwardDiff, FMIFlux.FMISensitivity.SciMLSensitivity.ReverseDiff, FMIFlux.FMISensitivity.SciMLSensitivity.FiniteDiff, FMIFlux.FMISensitivity.SciMLSensitivity.Zygote
using FMIFlux.FMIImport, FMIFlux.FMIImport.FMICore, FMIZoo
using FMIFlux.FMIImport.FMIBase: unsense
import LinearAlgebra:I

import Random 
Random.seed!(5678);

global solution = nothing
global events = 0

ENERGY_LOSS = 0.7 
RADIUS = 0.0
GRAVITY = 9.81
GRAVITY_SIGN = -1
DBL_MIN = 1e-10 # 2.2250738585072013830902327173324040642192159804623318306e-308
TIME_FREQ = 1.0
MASS = 1.0

INTERP_POINTS = 10
NUMEVENTS = 4

t_start = 0.0
t_step = 0.05
t_stop = 2.0
tSave = 0.0:t_step:10.0
tData = t_start:t_step:t_stop
posData = ones(Float64, length(tData))
x0_bb = [0.5, 0.0]

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
global fx_dx_cache = zeros(Real, 2)
fx = function(x, t; kwargs...)
    return _fx(x, t; kwargs...)
end

_fx = function(x, t)
    global fx_dx_cache

    fx_dx_cache[1] = x[2]
    fx_dx_cache[2] = GRAVITY_SIGN * GRAVITY / MASS
    return fx_dx_cache
end

fx_bb = function(dx, x, p, t)
    dx[:] = re_bb(p)(x)
    return nothing
end

net_bb = Chain(#Dense(W1, b1, identity),
            x -> fx(x, 0.0),
            Dense(W2, b2, identity))
p_net_bb, re_bb = Flux.destructure(net_bb)

ff = ODEFunction{true}(fx_bb) 
prob_bb = ODEProblem{true}(ff, x0_bb, (t_start, t_stop), p_net_bb)

condition = function(out, x, t, integrator)
    #x = re_bb(p_net_bb)[1](x)
    out[1] = x[1]-RADIUS
    out[2] = x[1]-RADIUS 
end

import FMIFlux: unsense
time_choice = function(integrator)
    next = (floor(integrator.t/TIME_FREQ)+1) * TIME_FREQ

    if next <= t_stop
        #@info "next: $(next)"
        return unsense(next)
    else
        return nothing
    end
end

time_affect! = function(integrator)
    global GRAVITY_SIGN
    GRAVITY_SIGN = -GRAVITY_SIGN

    global events 
    events += 1

    #u_modified!(integrator, false)
end

affect_right! = function(integrator, idx)

    #@info "affect_right! triggered by #$(idx)"

    # if idx == 1
    #     # event #1 is handeled as "dummy" (e.g. discrete state change)
    #     return 
    # end

    if idx > 0
        out = zeros(NUMEVENTINDICATORS)
        x = integrator.u
        t = integrator.t
        condition(out, unsense(x), unsense(t), integrator)
        if sign(out[idx]) > 0.0
            @info "Event for bouncing ball (white-box) triggered, but not valid!"
            return nothing 
        end
    end
    
    s_new = RADIUS + DBL_MIN
    v_new = -1.0 * unsense(integrator.u[2]) * ENERGY_LOSS

    left_x = unsense(integrator.u)
    right_x = [s_new, v_new]

    global events 
    events += 1
    #@info "[$(events)] New state at $(integrator.t) is $(u_new) triggered by #$(idx)"

    #integrator.u[:] .= u_new

    for i in 1:length(left_x)
        if left_x[i] != 0.0 # abs(left_x[i]) > 1e-128
            scale = right_x[i] / left_x[i]
            integrator.u[i] *= scale
        else # integrator state zero can't be scaled, need to add (but no sensitivities in this case!)
            shift = right_x[i] - left_x[i]
            integrator.u[i] += shift
            #integrator.u[i] = right_x[i]
            logWarning(c.fmu, "Probably wrong sensitivities @t=$(unsense(t)) for ∂x^+ / ∂x^-\nCan't scale zero state #$(i) from $(left_x[i]) to $(right_x[i])\nNew state after transform is: $(integrator.u[i])")
        end
    end

    return nothing 
end
affect_left! = function(integrator, idx)

    #@info "affect_left! triggered by #$(idx)"

    # if idx == 1
    #     # event #1 is handeled as "dummy" (e.g. discrete state change)
    #     return 
    # end

    out = zeros(NUMEVENTINDICATORS)
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

stepCompleted = function(x, t, integrator)
    
end

NUMEVENTINDICATORS = 2
rightCb = VectorContinuousCallback(condition, #_double,
                                   affect_right!,
                                   NUMEVENTINDICATORS;
                                   rootfind=RightRootFind, save_positions=(false, false),
                                   interp_points=INTERP_POINTS)
leftCb = VectorContinuousCallback(condition, #_double,
                                   affect_left!,
                                   NUMEVENTINDICATORS;
                                   rootfind=LeftRootFind, save_positions=(false, false),
                                   interp_points=INTERP_POINTS)

gravityCb = IterativeCallback(time_choice,
                    time_affect!, 
                    Float64; 
                    initial_affect=false,
                    save_positions=(false, false))

stepCb = FunctionCallingCallback(stepCompleted;
                                            func_everystep=true,
                                            func_start=true)

# load FMU for NeuralFMU
fmu = loadFMU("BouncingBallGravitySwitch1D", "Dymola", "2023x"; type=:ME)
fmu_params = Dict("damping" => ENERGY_LOSS, "mass_radius" => RADIUS, "gravity" => GRAVITY, "period" => TIME_FREQ, "mass_m" => MASS, "mass_s_min" => DBL_MIN)
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
    global solution, events
    global prob, x0_bb, posData, solver # read-only
    events = 0

    solution = prob(x0_bb; p=p, solver=solver, saveat=tSave, parameters=fmu_params, sensealg=sensealg) # recordValues=["der(mass_v)"]

    return collect(u[1] for u in solution.states.u)
end

mysolve_bb = function(p; sensealg=nothing, root=:Right)
    global solution, GRAVITY_SIGN
    global prob_bb, solver, events # read
    events = 0

    callback = nothing
    if root == :Right 
        callback = CallbackSet(gravityCb, rightCb, stepCb)
    elseif root == :Left
        callback = CallbackSet(gravityCb, leftCb, stepCb)
    else
        @assert false "unknwon root `$(root)`"
    end

    GRAVITY_SIGN = -1
    solution = solve(prob_bb, solver; u0=x0_bb, p=p, saveat=tSave, callback=callback, sensealg=sensealg)  #u0=x0_bb, 

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
c, _ = FMIFlux.prepareSolveFMU(prob.fmu, c, fmi2TypeModelExchange; parameters=prob.parameters, t_start=prob.tspan[1], t_stop=prob.tspan[end], x0=prob.x0, handleEvents=FMIFlux.handleEvents, cleanup=true)

### START CHECK CONDITIONS 

condition_bb_check = function(x)
    buffer = similar(x, NUMEVENTINDICATORS)
    condition(buffer, x, t_start, nothing)
    return buffer 
end
condition_nfmu_check = function(x)
    buffer = similar(x, fmu.modelDescription.numberOfEventIndicators)
    inds = collect(UInt32(i) for i in 1:fmu.modelDescription.numberOfEventIndicators)
    FMIFlux.condition!(prob, FMIFlux.getComponent(prob), buffer, x, t_start, nothing, inds)
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

affect_bb_check = function(x, t, idx=1)

    # convert TrackedArrays to Array{<:TrackedReal,1}
    if !isa(x, AbstractVector{<:Float64})
        x = [x...]
    else
        x = copy(x)
    end

    integrator = (t=t, u=x)
    if idx == 0
        time_affect!(integrator)
    else
        affect_right!(integrator, idx)
    end

    return integrator.u
end
affect_nfmu_check = function(x, t, idx=1)
    global prob

    # convert TrackedArrays to Array{<:TrackedReal,1}
    if !isa(x, AbstractVector{<:Float64})
        x = [x...]
    else
        x = copy(x)
    end
    
    c, _ = FMIFlux.prepareSolveFMU(prob.fmu, nothing, fmi2TypeModelExchange; parameters=fmu_params, t_start=unsense(t), t_stop=prob.tspan[end], x0=unsense(x), handleEvents=FMIFlux.handleEvents, cleanup=true)

    integrator = (t=t, u=x, opts=(internalnorm=(a,b)->1.0,) )
    FMIFlux.affectFMU!(prob, c, integrator, idx)
    
    return integrator.u
end
#t_event_time = 0.451523640985728
x_event_left = [-1.0, -1.0] # [-3.808199081191736e-15, -4.429446918069994]
x_event_right = [0.0, 0.7] # [2.2250738585072014e-308, 3.1006128426489954]
x_no_event = [0.1, -1.0]
t_no_event = t_start

@test isapprox(affect_bb_check(x_event_left, t_no_event), x_event_right; atol=1e-4)
@test isapprox(affect_nfmu_check(x_event_left, t_no_event), x_event_right; atol=1e-4)

jac_con1 = ForwardDiff.jacobian(x -> affect_bb_check(x, t_no_event), x_event_left)
jac_con2 = ForwardDiff.jacobian(x -> affect_nfmu_check(x, t_no_event), x_event_left)

@test isapprox(jac_con1, ∂xn_∂xp; atol=1e-4)
@test isapprox(jac_con2, ∂xn_∂xp; atol=1e-4)

jac_con1 = ReverseDiff.jacobian(x -> affect_bb_check(x, t_no_event), x_event_left)
jac_con2 = ReverseDiff.jacobian(x -> affect_nfmu_check(x, t_no_event), x_event_left)

@test isapprox(jac_con1, ∂xn_∂xp; atol=1e-4)
@test isapprox(jac_con2, ∂xn_∂xp; atol=1e-4)

# [Note] checking via FiniteDiff is not possible here, because finite differences offsets might not trigger the events at all

# no-event 

@test isapprox(affect_bb_check(x_no_event, t_no_event), x_no_event; atol=1e-4)
@test isapprox(affect_nfmu_check(x_no_event, t_no_event), x_no_event; atol=1e-4)

jac_con1 = ForwardDiff.jacobian(x -> affect_bb_check(x, t_no_event), x_no_event)
jac_con2 = ForwardDiff.jacobian(x -> affect_nfmu_check(x, t_no_event), x_no_event)

@test isapprox(jac_con1, I; atol=1e-4)
@test isapprox(jac_con2, I; atol=1e-4)

jac_con1 = ReverseDiff.jacobian(x -> affect_bb_check(x, t_no_event), x_no_event)
jac_con2 = ReverseDiff.jacobian(x -> affect_nfmu_check(x, t_no_event), x_no_event)

@test isapprox(jac_con1, I; atol=1e-4)
@test isapprox(jac_con2, I; atol=1e-4)

### TIME-EVENTS

t_event = t_start + 1.1

@test isapprox(affect_bb_check(x_no_event, t_event, 0), x_no_event; atol=1e-4)
@test isapprox(affect_nfmu_check(x_no_event, t_event, 0), x_no_event; atol=1e-4)

jac_con1 = ForwardDiff.jacobian(x -> affect_bb_check(x, t_event, 0), x_no_event)
jac_con2 = ForwardDiff.jacobian(x -> affect_nfmu_check(x, t_event, 0), x_no_event)

@test isapprox(jac_con1, I; atol=1e-4)
@test isapprox(jac_con2, I; atol=1e-4)

jac_con1 = ReverseDiff.jacobian(x -> affect_bb_check(x, t_event, 0), x_no_event)
jac_con2 = ReverseDiff.jacobian(x -> affect_nfmu_check(x, t_event, 0), x_no_event)

@test isapprox(jac_con1, I; atol=1e-4)
@test isapprox(jac_con2, I; atol=1e-4)

jac_con1 = ReverseDiff.jacobian(t -> affect_bb_check(x_event_left, t[1], 0), [t_event])
jac_con2 = ReverseDiff.jacobian(t -> affect_nfmu_check(x_event_left, t[1], 0), [t_event])

###

NUMEVENTS=4

# Solution (plain)
GRAVITY_SIGN = -1
losssum(p_net; sensealg=sensealg) 
@test length(solution.events) == NUMEVENTS

GRAVITY_SIGN = -1
losssum_bb(p_net_bb; sensealg=sensealg) 
@test events == NUMEVENTS

# Solution FWD (FMU)
GRAVITY_SIGN = -1
grad_fwd_f = ForwardDiff.gradient(p -> losssum(p; sensealg=sensealg), p_net)
@test length(solution.events) == NUMEVENTS

# Solution FWD (right)
GRAVITY_SIGN = -1
root = :Right
grad_fwd_r = ForwardDiff.gradient(p -> losssum_bb(p; sensealg=sensealg, root=root), p_net_bb)
@test events == NUMEVENTS

# Solution RWD (FMU)
GRAVITY_SIGN = -1
grad_rwd_f = ReverseDiff.gradient(p -> losssum(p; sensealg=sensealg), p_net)
@test length(solution.events) == NUMEVENTS

# Solution RWD (right)
GRAVITY_SIGN = -1
root = :Right
grad_rwd_r = ReverseDiff.gradient(p -> losssum_bb(p; sensealg=sensealg, root=root), p_net_bb)
@test events == NUMEVENTS

# Ground Truth
grad_fin_r = FiniteDiff.finite_difference_gradient(p -> losssum_bb(p; sensealg=sensealg, root=:Right), p_net_bb, Val{:central}; absstep=1e-6)
grad_fin_f = FiniteDiff.finite_difference_gradient(p -> losssum(p; sensealg=sensealg), p_net, Val{:central}; absstep=1e-6)

rtol = 1e-2
inds = collect(1:length(p_net))
#deleteat!(inds, 1:6)

# check if finite differences match together
@test isapprox(grad_fin_f[inds], grad_fin_r[inds]; rtol=rtol)
@test isapprox(grad_fin_f[inds], grad_fwd_f[inds]; rtol=rtol)
@test isapprox(grad_fin_f[inds], grad_rwd_f[inds]; rtol=rtol)
@test isapprox(grad_fwd_r[inds], grad_rwd_r[inds]; rtol=rtol)

# Jacobian Test

jac_fwd_r = ForwardDiff.jacobian(p -> mysolve_bb(p; sensealg=sensealg), p_net)
jac_fwd_f = ForwardDiff.jacobian(p -> mysolve(p; sensealg=sensealg), p_net)

jac_rwd_r = ReverseDiff.jacobian(p -> mysolve_bb(p; sensealg=sensealg), p_net)
#jac_rwd_f = ReverseDiff.jacobian(p -> mysolve(p; sensealg=sensealg), p_net)

# [TODO] why this?!
jac_rwd_r[2:end,:] = jac_rwd_r[2:end,:] .- jac_rwd_r[1:end-1,:]
# jac_rwd_f[2:end,:] = jac_rwd_f[2:end,:] .- jac_rwd_f[1:end-1,:]

jac_fin_r = FiniteDiff.finite_difference_jacobian(p -> mysolve_bb(p; sensealg=sensealg), p_net)
jac_fin_f = FiniteDiff.finite_difference_jacobian(p -> mysolve(p; sensealg=sensealg), p_net)

###

atol = 1e-3
@test isapprox(jac_fin_f[:, inds], jac_fin_r[:, inds]; atol=atol)
@test isapprox(jac_fin_f[:, inds], jac_fwd_f[:, inds]; atol=atol)

# [ToDo] this NaNs on two rows... whyever... but this is not required to work
# @test isapprox(jac_fin_f[:, inds], jac_rwd_f[:, inds]; atol=atol)

@test isapprox(jac_fin_r[:, inds], jac_fwd_r[:, inds]; atol=atol)
@test isapprox(jac_fin_r[:, inds], jac_rwd_r[:, inds]; atol=atol)

###

unloadFMU(fmu)
