#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMIImport.FMIBase: assert_integrator_valid, isdual, istracked, issense, undual, unsense, unsense_copy, untrack, FMUSnapshot
import FMIImport: finishSolveFMU, handleEvents, prepareSolveFMU, snapshot_if_needed!, getSnapshot
import Optim
import FMIImport.FMIBase.ProgressMeter
import FMISensitivity.SciMLSensitivity.SciMLBase: CallbackSet, ContinuousCallback, ODESolution, ReturnCode, RightRootFind,
    VectorContinuousCallback, set_u!, terminate!, u_modified!, build_solution
import OrdinaryDiffEq: isimplicit, alg_autodiff
using FMISensitivity.ReverseDiff: TrackedArray
import FMISensitivity.SciMLSensitivity: InterpolatingAdjoint, ReverseDiffVJP, AutoForwardDiff
import ThreadPools
import FMIImport.FMIBase

using FMIImport.FMIBase.DiffEqCallbacks
using FMIImport.FMIBase.SciMLBase: ODEFunction, ODEProblem, solve
using FMIImport.FMIBase: fmi2ComponentState,
    fmi2ComponentStateContinuousTimeMode, fmi2ComponentStateError,
    fmi2ComponentStateEventMode, fmi2ComponentStateFatal,
    fmi2ComponentStateInitializationMode, fmi2ComponentStateInstantiated,
    fmi2ComponentStateTerminated, fmi2StatusOK, fmi2Type, fmi2TypeCoSimulation,
    fmi2TypeModelExchange, logError, logInfo, logWarning , fast_copy!
using FMISensitivity.SciMLSensitivity:
    ForwardDiffSensitivity, InterpolatingAdjoint, ReverseDiffVJP, ZygoteVJP
import DifferentiableEigen
import DifferentiableEigen.LinearAlgebra: I

import FMIImport.FMIBase: EMPTY_fmi2Real, EMPTY_fmi2ValueReference
import FMIImport.FMIBase
import FMISensitivity: NoTangent, ZeroTangent

DEFAULT_PROGRESS_DESCR = "Simulating ME-NeuralFMU ..."
DEFAULT_CHUNK_SIZE = 32

"""
The mutable struct representing an abstract (simulation mode unknown) NeuralFMU.
"""
abstract type NeuralFMU end

"""
Structure definition for a NeuralFMU, that runs in mode `Model Exchange` (ME).
"""
mutable struct ME_NeuralFMU{M, R} <: NeuralFMU

    model::M
    p::AbstractArray{<:Real}
    re::R
    solvekwargs

    re_model 
    re_p

    fmu::FMU

    tspan
    saveat
    saved_values
    recordValues
    solver

    valueStack

    customCallbacksBefore::Array
    customCallbacksAfter::Array

    x0::Union{Array{Float64}, Nothing}
    firstRun::Bool
    
    tolerance::Union{Real, Nothing}
    parameters::Union{Dict{<:Any, <:Any}, Nothing}
    
    modifiedState::Bool

    execution_start::Real

    condition_buffer::Union{AbstractArray{<:Real}, Nothing}

    snapshots::Bool

    function ME_NeuralFMU{M, R}(model::M, p::AbstractArray{<:Real}, re::R) where {M, R}
        inst = new()
        inst.model = model 
        inst.p = p 
        inst.re = re 
        inst.x0 = nothing
        inst.saveat = nothing

        inst.re_model = nothing
        inst.re_p = nothing

        inst.modifiedState = false

        # inst.startState = nothing 
        # inst.stopState = nothing
        # inst.startEventInfo = nothing 
        # inst.stopEventInfo = nothing

        inst.customCallbacksBefore = []
        inst.customCallbacksAfter = []

        inst.execution_start = 0.0

        inst.condition_buffer = nothing

        inst.snapshots = false
       
        return inst 
    end
end

"""
Structure definition for a NeuralFMU, that runs in mode `Co-Simulation` (CS).
"""
mutable struct CS_NeuralFMU{F, C} <: NeuralFMU
    model
    fmu::F
    
    tspan
    
    p::Union{AbstractArray{<:Real}, Nothing}
    re # restrucure function

    snapshots::Bool

    function CS_NeuralFMU{F, C}() where {F, C}
        inst = new{F, C}()

        inst.re = nothing
        inst.p = nothing

        inst.snapshots = false

        return inst
    end
end

function evaluateModel(nfmu::ME_NeuralFMU, c::FMU2Component, x; p=nfmu.p, t=c.default_t)
    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    # [ToDo]: Skip array check, e.g. by using a flag
    #if p !== nfmu.re_p || p != nfmu.re_p # || isnothing(nfmu.re_model)
    #    nfmu.re_p = p # fast_copy!(nfmu, :re_p, p)
    #    nfmu.re_model = nfmu.re(p)
    #end
    #return nfmu.re_model(x)

    @debug "evaluateModel(t=$(t)) [out-of-place dx]"

    #nfmu.p = p 
    c.default_t = t
    return nfmu.re(p)(x)
end

function evaluateModel(nfmu::ME_NeuralFMU, c::FMU2Component, dx, x; p=nfmu.p, t=c.default_t)
    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    # [ToDo]: Skip array check, e.g. by using a flag
    #if p !== nfmu.re_p || p != nfmu.re_p # || isnothing(nfmu.re_model)
    #    nfmu.re_p = p # fast_copy!(nfmu, :re_p, p)
    #    nfmu.re_model = nfmu.re(p)
    #end
    #dx[:] = nfmu.re_model(x)

    @debug "evaluateModel(t=$(t)) [in-place dx]"

    #nfmu.p = p 
    c.default_t = t
    dx[:] = nfmu.re(p)(x)

    return nothing
end

##### EVENT HANDLING START

function startCallback(integrator, nfmu::ME_NeuralFMU, c::Union{FMU2Component, Nothing}, t, writeSnapshot, readSnapshot)

    ignore_derivatives() do

        nfmu.execution_start = time()

        t = unsense(t)

        @assert t == nfmu.tspan[1] "startCallback(...): Called for non-start-point t=$(t)"
        
        c, x0 = prepareSolveFMU(nfmu.fmu, c, fmi2TypeModelExchange; parameters=nfmu.parameters, t_start=nfmu.tspan[1], t_stop=nfmu.tspan[end], tolerance=nfmu.tolerance, x0=nfmu.x0, handleEvents=FMIFlux.handleEvents, cleanup=true)
        
        if c.eventInfo.nextEventTime == t && c.eventInfo.nextEventTimeDefined == fmi2True
            @debug "Initial time event detected!"
        else
            @debug "No initial time events ..."
        end

        if nfmu.snapshots
            FMIBase.snapshot!(c.solution)
        end

        if !isnothing(writeSnapshot)
            FMIBase.update!(c, writeSnapshot)
        end

        if !isnothing(readSnapshot)
            @assert c == readSnapshot.instance "Snapshot instance mismatch, snapshot instance is $(readSnapshot.instance.compAddr), current component is $(c.compAddr)"
            # c = readSnapshot.instance 

            if t != readSnapshot.t
                logWarning(c.fmu, "Snapshot time mismatch, snapshot time = $(readSnapshot.t), but start time is $(t)")
            end

            @debug "ME_NeuralFMU: Applying snapshot..."
            FMIBase.apply!(c, readSnapshot; t=t)
            @debug "ME_NeuralFMU: Snapshot applied."
        end

    end

    return c
end

function stopCallback(nfmu::ME_NeuralFMU, c::FMU2Component, t)

    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    ignore_derivatives() do
        
        t = unsense(t)

        @assert t == nfmu.tspan[end] "stopCallback(...): Called for non-start-point t=$(t)"
    end

    return c
end

# Read next time event from fmu and provide it to the integrator 
function time_choice(nfmu::ME_NeuralFMU, c::FMU2Component, integrator, tStart, tStop)

    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @assert c.fmu.executionConfig.handleTimeEvents "time_choice(...) was called, but execution config disables time events.\nPlease open a issue."
    # assert_integrator_valid(integrator)

    # last call may be after simulation end
    if c == nothing
        return nothing
    end

    c.solution.evals_timechoice += 1

    if c.eventInfo.nextEventTimeDefined == fmi2True
        
        if c.eventInfo.nextEventTime >= tStart && c.eventInfo.nextEventTime <= tStop
            @debug "time_choice(...): At $(integrator.t) next time event announced @$(c.eventInfo.nextEventTime)s"
            return c.eventInfo.nextEventTime
        else
            # the time event is outside the simulation range!
            @debug "Next time event @$(c.eventInfo.nextEventTime)s is outside simulation time range ($(tStart), $(tStop)), skipping."
            return nothing 
        end
    else
        return nothing
    end
end

# [ToDo] for now, ReverseDiff (together with the rrule) seems to have a problem with the SubArray here (when `collect` it accesses array elements that are #undef), 
#        so I added an additional (single allocating) dispatch...
#        Type is ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}[#undef, #undef, #undef, ...]
function condition!(nfmu::ME_NeuralFMU, c::FMU2Component, out::AbstractArray{<:ReverseDiff.TrackedReal}, x, t, integrator, handleEventIndicators) 

    if !isassigned(out, 1)
        if isnothing(nfmu.condition_buffer)
            logInfo(nfmu.fmu, "There is currently an issue with the condition buffer pre-allocation, the buffer can't be overwritten by the generated rrule.\nBuffer is generated automatically.")
            @assert length(out) == length(handleEventIndicators) "Number of event indicators to handle ($(handleEventIndicators)) doesn't fit buffer size $(length(out))."
            nfmu.condition_buffer = zeros(eltype(out), length(out))
        elseif eltype(out) != eltype(nfmu.condition_buffer) || length(out) != length(nfmu.condition_buffer)
            nfmu.condition_buffer = zeros(eltype(out), length(out))
        end
        out[:] = nfmu.condition_buffer 
    end

    invoke(condition!, Tuple{ME_NeuralFMU, FMU2Component, Any,  Any, Any, Any, Any}, nfmu, c, out, x, t, integrator, handleEventIndicators)
    
    return nothing
end

function condition!(nfmu::ME_NeuralFMU, c::FMU2Component, out, x, t, integrator, handleEventIndicators) 

    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @assert c.state == fmi2ComponentStateContinuousTimeMode "condition!(...):\n" * FMICore.ERR_MSG_CONT_TIME_MODE

    # [ToDo] Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN.
    #        Basically only the layers from very top to FMU need to be evaluated here.

    prev_t = c.default_t 
    prev_ec = c.default_ec 
    prev_ec_idcs = c.default_ec_idcs

    c.default_t = t
    c.default_ec = out
    c.default_ec_idcs = handleEventIndicators
    
    evaluateModel(nfmu, c, x)
    # write back to condition buffer
    if (!isdual(out) && isdual(c.output.ec)) || (!istracked(out) && istracked(c.output.ec))
        out[:] = unsense(c.output.ec)
    else
        out[:] = c.output.ec # [ToDo] This seems not to be necessary, because of `c.default_ec = out`
    end

    # reset
    c.default_t = prev_t
    c.default_ec = prev_ec
    c.default_ec_idcs = prev_ec_idcs
    
    c.solution.evals_condition += 1

    @debug "condition!(...) -> [typeof=$(typeof(out))]\n$(unsense(out))"

    return nothing
end

global lastIndicator = nothing
global lastIndicatorX = nothing 
global lastIndicatorT = nothing
function conditionSingle(nfmu::ME_NeuralFMU, c::FMU2Component, index, x, t, integrator) 

    @assert c.state == fmi2ComponentStateContinuousTimeMode "condition(...):\n" * FMICore.ERR_MSG_CONT_TIME_MODE
    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    if c.fmu.handleEventIndicators != nothing && index ∉ c.fmu.handleEventIndicators
        return 1.0
    end

    global lastIndicator

    if lastIndicator == nothing || length(lastIndicator) != c.fmu.modelDescription.numberOfEventIndicators
        lastIndicator = zeros(c.fmu.modelDescription.numberOfEventIndicators)
    end

    # [ToDo] Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    c.default_t = t
    c.default_ec = lastIndicator
    evaluateModel(nfmu, c, x)
    c.default_t = -1.0
    c.default_ec = EMPTY_fmi2Real
    
    c.solution.evals_condition += 1
    
    return lastIndicator[index]
end

function smoothmax(vec::AbstractVector; alpha=0.5)
    dividend = 0.0
    divisor = 0.0
    e = Float64(ℯ)
    for x in vec
        dividend += x * e^(alpha * x)
        divisor += e^(alpha * x)
    end
    return dividend/divisor
end

function smoothmax(a, b; kwargs...)
    return smoothmax([a, b]; kwargs...)
end

# [ToDo] Check, that the new determined state is the right root of the event instant!
function f_optim(x, nfmu::ME_NeuralFMU, c::FMU2Component, right_x_fmu, idx, sign::Real, out, indicatorValue, handleEventIndicators; _unsense::Bool=false)

    prev_ec = c.default_ec 
    prev_ec_idcs = c.default_ec_idcs
    prev_y_refs = c.default_y_refs
    prev_y = c.default_y

    #@info "\ndx: $(c.default_dx)\n x: $(x)"
    
    c.default_ec = out
    c.default_ec_idcs = handleEventIndicators
    c.default_y_refs = c.fmu.modelDescription.stateValueReferences
    c.default_y = zeros(typeof(x[1]), length(c.fmu.modelDescription.stateValueReferences))
    
    evaluateModel(nfmu, c, x; p=unsense(nfmu.p))
    # write back to condition buffer
    # if (!isdual(out) && isdual(c.output.ec)) || (!istracked(out) && istracked(c.output.ec))
    #     @assert false "Type missmatch! Can't propagate sensitivities!"
    #     out[:] = unsense(c.output.ec)
    # else
    #     out[:] = c.output.ec # [ToDo] This seems not to be necessary, because of `c.default_ec = out`
    # end

    # reset
    c.default_ec = prev_ec
    c.default_ec_idcs = prev_ec_idcs
    c.default_y_refs = prev_y_refs
    c.default_y = prev_y
    



    # propagete the new state-guess `x` through the NeuralFMU
    
    #condition!(nfmu, c, buffer, x, c.t, nothing, handleEventIndicators) 

    ec = c.output.ec[idx]
    y = c.output.y

    #@info "\nec: $(ec)\n-> $(unsense(ec))\ny: $(y)\n-> $(unsense(y))"

    errorIndicator = Flux.Losses.mae(indicatorValue, ec) + smoothmax(-sign*ec*1000.0, 0.0)
    # if errorIndicator > 0.0 
    #     errorIndicator = max(errorIndicator, 1.0)
    # end
    errorState = Flux.Losses.mae(right_x_fmu, y)

    #@info "ErrorState: $(errorState) | ErrorIndicator: $(errorIndicator)"

    ret =  errorState + errorIndicator

    # if _unsense
    #     ret = unsense(ret)
    # end

    return ret 
end

function sampleStateChangeJacobian(nfmu, c, left_x, right_x, t, idx::Integer; step = 1e-8)

    c.solution.evals_∂xr_∂xl += 1
    
    numStates = length(left_x)
    jac = zeros(numStates, numStates)
    
    # first, jump to before the event instance
    # if length(c.solution.snapshots) > 0 # c.t != t 
    #     sn = getSnapshot(c.solution, t)
    #     FMIBase.apply!(c, sn; x_c=left_x, t=t)
    #     #@info "[d] Set snapshot @ t=$(t) (sn.t=$(sn.t))"
    # end
    # indicator_sign = idx > 0 ? sign(fmi2GetEventIndicators(c)[idx]) : 1.0

    # ONLY A TEST
    new_left_x = copy(left_x)
    if length(c.solution.snapshots) > 0 # c.t != t 
        sn = getSnapshot(c.solution, t)
        FMIBase.apply!(c, sn; x_c=new_left_x, t=t)
        #@info "[?] Set snapshot @ t=$(t) (sn.t=$(sn.t))"
    end
    new_right_x = stateChange!(nfmu, c, new_left_x, t, idx; snapshots=false)
    statesChanged = (c.eventInfo.valuesOfContinuousStatesChanged == fmi2True)
    
    # [ToDo: these tests should be included, but will drastically fail on FMUs with no support for get/setState]
    # @assert statesChanged "Can't reproduce event (statesChanged)!" 
    # @assert left_x == new_left_x "Can't reproduce event (left_x)!"
    # @assert right_x == new_right_x "Can't reproduce event (right_x)!"

    at_least_one_state_change = false
    
    for i in 1:numStates
        
        #new_left_x[:] .= left_x
        new_left_x = copy(left_x)
        new_left_x[i] += step

        # first, jump to before the event instance
        if length(c.solution.snapshots) > 0 # c.t != t 
            sn = getSnapshot(c.solution, t)
            FMIBase.apply!(c, sn; x_c=new_left_x, t=t)
            #@info "[e] Set snapshot @ t=$(t) (sn.t=$(sn.t))"
        end
        # [ToDo] Don't check if event was handled via event-indicator, because there is no guarantee that it is reset (like for the bouncing ball)
        #        to match the sign from before the event! Better check if FMU detects a new event!
        # fmi2EnterEventMode(c)
        # handleEvents(c)
        new_right_x = stateChange!(nfmu, c, new_left_x, t, idx; snapshots=false)
        statesChanged = (c.eventInfo.valuesOfContinuousStatesChanged == fmi2True)
        at_least_one_state_change = statesChanged || at_least_one_state_change
        #new_indicator_sign = idx > 0 ? sign(fmi2GetEventIndicators(c)[idx]) : 1.0
        #@info "Sample P: t:$(t)   $(new_left_x) -> $(new_right_x)"

        grad = (new_right_x .- right_x) ./ step # (left_x .- new_left_x)
        
        # choose other direction
        if !statesChanged 
            #@info "New_indicator sign is $(new_indicator_sign) (should be $(indicator_sign)), retry..."
            #new_left_x[:] .= left_x
            new_left_x = copy(left_x)
            new_left_x[i] -= step
            
            if length(c.solution.snapshots) > 0 # c.t != t 
                sn = getSnapshot(c.solution, t)
                FMIBase.apply!(c, sn; x_c=new_left_x, t=t)
                #@info "[e] Set snapshot @ t=$(t) (sn.t=$(sn.t))"
            end
            #fmi2EnterEventMode(c)
            #handleEvents(c)
            new_right_x = stateChange!(nfmu, c, new_left_x, t, idx; snapshots=false)
            statesChanged = (c.eventInfo.valuesOfContinuousStatesChanged == fmi2True)
            at_least_one_state_change = statesChanged || at_least_one_state_change
            #new_indicator_sign = idx > 0 ? sign(fmi2GetEventIndicators(c)[idx]) : 1.0

            #@info "Sample N: t:$(t)   $(new_left_x) -> $(new_right_x)"

            if statesChanged
                grad = (new_right_x .- right_x) ./ -step # (left_x .- new_left_x)
            else
                grad = (right_x .- right_x) # ... so zero, this state is not sensitive at all!
            end
            
        end

        # if length(c.solution.snapshots) > 0 # c.t != t 
        #     sn = getSnapshot(c.solution, t)
        #     FMIBase.apply!(c, sn; x_c=new_left_x, t=t)
        #     #@info "[e] Set snapshot @ t=$(t) (sn.t=$(sn.t))"
        # end

        # new_right_x = stateChange!(nfmu, c, new_left_x, t, idx; snapshots=false)

        # [ToDo] check if the SAME event indicator was triggered!

        #@info "t=$(t) idx=$(idx)\n    left_x: $(left_x)   ->       right_x: $(right_x)   [$(indicator_sign)]\nnew_left_x: $(new_left_x)   ->   new_right_x: $(new_right_x)   [$(new_indicator_sign)]"

        

        jac[i,:] = grad
    end

    @assert at_least_one_state_change "Sampling state change jacobian failed, can't find another state that triggers the event!"


    # finally, jump back to the correct FMU state 
    # if length(c.solution.snapshots) > 0 # c.t != t 
    #     @info "Reset snapshot @ t = $(t)"
    #     sn = getSnapshot(c.solution, t)
    #     FMIBase.apply!(c, sn; x_c=left_x, t=t)
    # end
    # stateChange!(nfmu, c, left_x, t, idx)
    if length(c.solution.snapshots) > 0 
        #@info "Reset exact snapshot @t=$(t)"
        sn = getSnapshot(c.solution, t; exact=true)
        if !isnothing(sn)
            FMIBase.apply!(c, sn; x_c=left_x, t=t)
        end
    end
    
    #@info "Jac:\n$(jac)"
    #@assert isapprox(jac, [0.0 0.0; 0.0 -0.7]; atol=1e-4) "Jac missmatch, is $(jac)"

    return jac
end

function is_integrator_sensitive(integrator)
    return istracked(integrator.u) || istracked(integrator.t) || isdual(integrator.u) || isdual(integrator.t)
end

function stateChange!(nfmu, c, left_x::AbstractArray{<:Float64}, t::Float64, idx; snapshots=nfmu.snapshots)

    # unpack references 
    # if typeof(cRef) != UInt64
    #     cRef = UInt64(cRef)
    # end
    # c = unsafe_pointer_to_objref(Ptr{Nothing}(cRef))
    # if typeof(nfmuRef) != UInt64
    #     nfmuRef = UInt64(nfmuRef)
    # end
    # nfmu = unsafe_pointer_to_objref(Ptr{Nothing}(nfmuRef))
    # unpack references  done 

    # if length(c.solution.snapshots) > 0 # c.t != t 
    #     sn = getSnapshot(c.solution, t)
    #     @info "[x] Set snapshot @ t=$(t) (sn.t=$(sn.t))"
    #     FMIBase.apply!(c, sn; x_c=left_x, t=t)
    # end

    fmi2EnterEventMode(c)
    handleEvents(c)

    #snapshots = true # nfmu.snapshots || snapshotsNeeded(nfmu, integrator)

    # ignore_derivatives() do
    #     if idx == 0
    #         time_affect!(integrator)
    #     else
    #         #affect_right!(integrator, idx)
    #     end
    # end

    right_x = left_x 

    if c.eventInfo.valuesOfContinuousStatesChanged == fmi2True

        ignore_derivatives() do 
            if idx == 0
                @debug "affectFMU!(_, _, $idx): NeuralFMU time event with state change.\nt = $(t)\nleft_x = $(left_x)"
            else
                @debug "affectFMU!(_, _, $idx): NeuralFMU state event with state change by indicator $(idx).\nt = $(t)\nleft_x = $(left_x)"
            end
        end

        right_x_fmu = fmi2GetContinuousStates(c) # the new FMU state after handled events

        # if there is an ANN above the FMU, propaget FMU state through top ANN by optimization
        if nfmu.modifiedState 
            before = fmi2GetEventIndicators(c)
            buffer = copy(before)
            handleEventIndicators = Vector{UInt32}(collect(i for i in 1:length(nfmu.fmu.modelDescription.numberOfEventIndicators)))

            _f(_x)   = f_optim(_x, nfmu, c, right_x_fmu, idx, sign(before[idx]), buffer, before[idx], handleEventIndicators; _unsense=true)
            _f_g(_x) = f_optim(_x, nfmu, c, right_x_fmu, idx, sign(before[idx]), buffer, before[idx], handleEventIndicators; _unsense=false)

            function _g!(G, x)
                #if istracked(integrator.u) 
                #    ReverseDiff.gradient!(G, _f_g, x)
                #else # if isdual(integrator.u) 
                    ForwardDiff.gradient!(G, _f_g, x)
                # else
                #     @assert false "Unknown AD framework! -> $(typeof(integrator.u[1]))"
                #end

                #@info "G: $(G)"
            end

            result = Optim.optimize(_f, _g!, left_x, Optim.BFGS())
            right_x = Optim.minimizer(result)

            after = fmi2GetEventIndicators(c)
            if sign(before[idx]) != sign(after[idx])
                logError(nfmu.fmu, "Eventhandling failed,\nRight state: $(right_x)\nRight FMU state: $(right_x_fmu)\nIndicator (bef.): $(before[idx])\nIndicator (aft.): $(after[idx])")
            end
            
        else # if there is no ANN above, then:
            right_x = right_x_fmu
        end

    else

        ignore_derivatives() do 
            if idx == 0
                @debug "affectFMU!(_, _, $idx): NeuralFMU time event without state change.\nt = $(t)\nleft_x = $(left_x)"
            else
                @debug "affectFMU!(_, _, $idx): NeuralFMU state event without state change by indicator $(idx).\nt = $(t)\nleft_x = $(left_x)"
            end
        end

        # [Note] enabling this causes serious issues with time events! (wrong sensitivities!)
        # u_modified!(integrator, false)
    end

    if snapshots 
        s = snapshot_if_needed!(c.solution, t)
        # if !isnothing(s)
        #     @info "Add snapshot @t=$(s.t)"
        # end
    end

    # [ToDo] This is only correct, if every state is only depenent on itself.
    # This should only be done in the frule/rrule, the actual affect should do a hard "set state"
    #logWarning(c.fmu, "Before: integrator.u = $(integrator.u)")

    # if nfmu.fmu.executionConfig.isolatedStateDependency
    #     for i in 1:length(left_x)
    #         if abs(left_x[i]) > 1e-16 # left_x[i] != 0.0 # 
    #             scale = right_x[i] / left_x[i]
    #             integrator.u[i] *= scale
    #         else # integrator state zero can't be scaled, need to add (but no sensitivities in this case!)
    #             shift = right_x[i] - left_x[i]
    #             integrator.u[i] += shift
    #             #integrator.u[i] = right_x[i]
    #             #logWarning(c.fmu, "Probably wrong sensitivities @t=$(unsense(t)) for ∂x^+ / ∂x^-\nCan't scale zero state #$(i) from $(left_x[i]) to $(right_x[i])\nNew state after transform is: $(integrator.u[i])")
    #         end
    #     end
    # else
    #     integrator.u[:] = right_x
    # end

    return right_x
end

# Handles the upcoming event
function affectFMU!(nfmu::ME_NeuralFMU, c::FMU2Component, integrator, idx)

    @debug "affectFMU!"
    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    # assert_integrator_valid(integrator)

    @assert c.state == fmi2ComponentStateContinuousTimeMode "affectFMU!(...):\n" * FMICore.ERR_MSG_CONT_TIME_MODE

    # [NOTE] Here unsensing is OK, because we just want to reset the FMU to the correct state!
    #        The values come directly from the integrator and are NOT function arguments!
    t = unsense(integrator.t)
    left_x = unsense_copy(integrator.u)
    right_x = nothing
    
    ignore_derivatives() do

        # if snapshots && length(c.solution.snapshots) > 0 
        #     sn = getSnapshot(c.solution, t)
        #     FMIBase.apply!(c, sn)
        # end

    #if c.x != left_x
        # capture status of `force`    
        mode = c.force
        c.force = true

        # there are fx-evaluations before the event is handled, reset the FMU state to the current integrator step
        evaluateModel(nfmu, c, left_x; t=t) # evaluate NeuralFMU (set new states)
        # [NOTE] No need to reset time here, because we did pass a event instance! 
        #        c.default_t = -1.0

        c.force = mode
    #end
    end

    integ_sens = nfmu.snapshots

    right_x = stateChange!(nfmu, c, left_x, t, idx)
    
    # sensitivities needed
    if integ_sens
        jac = I 

        if c.eventInfo.valuesOfContinuousStatesChanged == fmi2True
            jac = sampleStateChangeJacobian(nfmu, c, left_x, right_x, t, idx)
        end

        VJP = jac * integrator.u 
        #tgrad = tvec .* integrator.t
        staticOff = right_x .- unsense(VJP) # .- unsense(tgrad)

        # [ToDo] add (sampled) time gradient
        integrator.u[:] = staticOff + VJP # + tgrad
    else
        integrator.u[:] = right_x
    end

    #@info "affect right_x = $(right_x)"
    
    # [Note] enabling this causes serious issues with time events! (wrong sensitivities!)
    # u_modified!(integrator, true)

    if c.eventInfo.nominalsOfContinuousStatesChanged == fmi2True
        # [ToDo] Do something with that information, e.g. use for FiniteDiff sampling step size determination
        x_nom = fmi2GetNominalsOfContinuousStates(c)
    end

    ignore_derivatives() do
        if idx != -1
            _left_x = left_x
            _right_x = isnothing(right_x) ? _left_x : unsense_copy(right_x)

            #@assert c.eventInfo.valuesOfContinuousStatesChanged == (_left_x != _right_x) "FMU says valuesOfContinuousStatesChanged $(c.eventInfo.valuesOfContinuousStatesChanged), but states say different!"
            e = FMUEvent(unsense(t), UInt64(idx), _left_x, _right_x)
            push!(c.solution.events, e)
        end

        # calculates state events per second
        pt = t-nfmu.tspan[1]
        ne = 0 
        for event in c.solution.events
            #if t - event.t < pt 
                if event.indicator > 0 # count only state events
                    ne += 1
                end
            #end
        end
        ratio = ne / pt
       
        if ne >= 100 && ratio > c.fmu.executionConfig.maxStateEventsPerSecond
            logError(c.fmu, "Event chattering detected $(round(Integer, ratio)) state events/s (allowed are $(c.fmu.executionConfig.maxStateEventsPerSecond)), aborting at t=$(t) (rel. t=$(pt)) at state event $(ne):")
            for i in 1:c.fmu.modelDescription.numberOfEventIndicators
                num = 0
                for e in c.solution.events
                    if e.indicator == i
                        num += 1 
                    end 
                end
                if num > 0
                    logError(c.fmu, "\tEvent indicator #$(i) triggered $(num) ($(round(num/ne*100.0; digits=1))%)")
                end
            end

            terminate!(integrator)
        end
    end

    c.solution.evals_affect += 1

    return nothing
end

# Does one step in the simulation.
function stepCompleted(nfmu::ME_NeuralFMU, c::FMU2Component, x, t, integrator, tStart, tStop)

    # assert_integrator_valid(integrator)

    # [Note] enabling this causes serious issues with time events! (wrong sensitivities!)
    # u_modified!(integrator, false)

    c.solution.evals_stepcompleted += 1

    # if snapshots
    #     FMIBase.snapshot!(c.solution)
    # end

    if !isnothing(c.progressMeter)
        t = unsense(t)
        dt = unsense(integrator.t) - unsense(integrator.tprev)
        events = length(c.solution.events)
        steps = c.solution.evals_stepcompleted
        simLen = tStop-tStart
        c.progressMeter.desc = "t=$(roundToLength(t, 10))s | Δt=$(roundToLength(dt, 10))s | STPs=$(steps) | EVTs=$(events) |"
        #@info "$(tStart)   $(tStop)   $(t)"

        if simLen > 0.0
            ProgressMeter.update!(c.progressMeter, floor(Integer, 1000.0*(t-tStart)/simLen) )
        end
    end

    if c != nothing
        (status, enterEventMode, terminateSimulation) = fmi2CompletedIntegratorStep(c, fmi2True)

        if terminateSimulation == fmi2True
            logError(c.fmu, "stepCompleted(...): FMU requested termination!")
        end

        if enterEventMode == fmi2True
            affectFMU!(nfmu, c, integrator, -1)
        end

        @debug "Step completed at $(unsense(t)) with $(unsense(x))"
    end

    # assert_integrator_valid(integrator)
end

# [ToDo] (1) This must be in-place 
#        (2) getReal must be replaced with the inplace getter within c(...)
#        (3) remove unsense to determine save value sensitivities
# save FMU values 
function saveValues(nfmu::ME_NeuralFMU, c::FMU2Component, recordValues, _x, _t, integrator)

    t = unsense(_t) 
    x = unsense(_x)

    c.solution.evals_savevalues += 1

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    evaluateModel(nfmu, c, x; t=t) # evaluate NeuralFMU (set new states)
   
    values = fmi2GetReal(c, recordValues)

    @debug "Save values @t=$(t)\nintegrator.t=$(unsense(integrator.t))\n$(values)"

    # Todo set inputs
    return (values...,)
end

function saveEigenvalues(nfmu::ME_NeuralFMU, c::FMU2Component, _x, _t, integrator, sensitivity::Symbol)

    @assert c.state == fmi2ComponentStateContinuousTimeMode "saveEigenvalues(...):\n" * FMICore.ERR_MSG_CONT_TIME_MODE

    c.solution.evals_saveeigenvalues += 1

    A = nothing
    if sensitivity == :ForwardDiff
        A = ForwardDiff.jacobian(x -> evaluateModel(nfmu, c, x; t=_t), _x) # TODO: chunk_size!
    elseif sensitivity == :ReverseDiff 
        A = ReverseDiff.jacobian(x -> evaluateModel(nfmu, c, x; t=_t), _x)
    elseif sensitivity == :Zygote 
        A = Zygote.jacobian(x -> evaluateModel(nfmu, c, x; t=_t), _x)[1]
    elseif sensitivity == :none
        A = ForwardDiff.jacobian(x -> evaluateModel(nfmu, c, x; t=_t), unsense(_x))
    end
    eigs, _ = DifferentiableEigen.eigen(A)

    return (eigs...,)
end

function fx(nfmu::ME_NeuralFMU,
    c::FMU2Component,
    dx,#::Array{<:Real},
    x,#::Array{<:Real},
    p,#::Array,
    t)#::Real) 

    if isnothing(c)
        # this should never happen!
        @warn "fx() called without allocated FMU instance!"
        return zeros(length(x))
    end

    ############

    evaluateModel(nfmu, c, dx, x; p=p, t=t)

    ignore_derivatives() do
        c.solution.evals_fx_inplace += 1
    end 

    return dx
end

function fx(nfmu::ME_NeuralFMU,
    c::FMU2Component,
    x,#::Array{<:Real},
    p,#::Array,
    t)#::Real) 
    
    if c === nothing
        # this should never happen!
        return zeros(length(x))
    end

    ignore_derivatives() do
        c.solution.evals_fx_outofplace += 1
    end
    
    return evaluateModel(nfmu, c, x; p=p, t=t)
end

##### EVENT HANDLING END

"""
Constructs a ME-NeuralFMU where the FMU is at an arbitrary location inside of the NN.

# Arguments
    - `fmu` the considered FMU inside the NN 
    - `model` the NN topology (e.g. Flux.chain)
    - `tspan` simulation time span
    - `solver` an ODE Solver (default=`nothing`, heurisitically determine one) 

# Keyword arguments
    - `recordValues` additionally records internal FMU variables 
"""
function ME_NeuralFMU(fmu::FMU2, 
                      model, 
                      tspan, 
                      solver=nothing; 
                      recordValues = nothing, 
                      saveat=nothing,
                      solvekwargs...)

    if !is64(model)
        model = convert64(model)
        logInfo(fmu, "Model is not Float64, but this is necessary for (Neural)FMUs.\nModel parameters are automatically converted to Float64.")
    end
    
    p, re = Flux.destructure(model)
    nfmu = ME_NeuralFMU{typeof(model), typeof(re)}(model, p, re)

    ######

    nfmu.fmu = fmu

    nfmu.saved_values = nothing

    nfmu.recordValues = prepareValueReference(fmu, recordValues)

    nfmu.tspan = tspan
    nfmu.solver = solver
    nfmu.saveat = saveat
    nfmu.solvekwargs = solvekwargs
    nfmu.parameters = nothing

    ######
    
    nfmu
end

"""
Constructs a CS-NeuralFMU where the FMU is at an arbitrary location inside of the ANN.

# Arguents
    - `fmu` the considered FMU inside the ANN 
    - `model` the ANN topology (e.g. Flux.Chain)
    - `tspan` simulation time span

# Keyword arguments
    - `recordValues` additionally records FMU variables 
"""
function CS_NeuralFMU(fmu::FMU2,
                      model, 
                      tspan; 
                      recordValues=[])

    if !is64(model)
        model = convert64(model)
        logInfo(fmu, "Model is not Float64, but this is necessary for (Neural)FMUs.\nModel parameters are automatically converted to Float64.")
    end

    nfmu = CS_NeuralFMU{FMU2, FMU2Component}()

    nfmu.fmu = fmu
    nfmu.model = model 
    nfmu.tspan = tspan

    nfmu.p, nfmu.re = Flux.destructure(nfmu.model)
    
    return nfmu
end

function CS_NeuralFMU(fmus::Vector{<:FMU2},
                      model, 
                      tspan; 
                      recordValues=[])

    if !is64(model)
        model = convert64(model)
        for fmu in fmus
            logInfo(fmu, "Model is not Float64, but this is necessary for (Neural)FMUs.\nModel parameters are automatically converted to Float64.")
        end
    end

    nfmu = CS_NeuralFMU{Vector{FMU2}, Vector{FMU2Component} }()
    
    nfmu.fmu = fmus
    nfmu.model = model 
    nfmu.tspan = tspan

    nfmu.p, nfmu.re = Flux.destructure(nfmu.model)
   
    return nfmu
end

function checkExecTime(integrator, nfmu::ME_NeuralFMU, c, max_execution_duration::Real)
    dist = max(nfmu.execution_start + max_execution_duration - time(), 0.0)
    
    if dist <= 0.0
        logInfo(nfmu.fmu, "Reached max execution duration ($(max_execution_duration)), terminating integration ...")
        terminate!(integrator)
    end

    return 1.0
end

function getInstance(nfmu::NeuralFMU)
    return hasCurrentInstance(nfmu.fmu) ? getCurrentInstance(nfmu.fmu) : nothing
end

# ToDo: Separate this: NeuralFMU creation and solving!
"""
    
    nfmu(x_start, tspan; kwargs)

Evaluates the ME_NeuralFMU `nfmu` in the timespan given during construction or in a custom timespan from `t_start` to `t_stop` for a given start state `x_start`.

# Keyword arguments

[ToDo]
"""
function (nfmu::ME_NeuralFMU)(x_start::Union{Array{<:Real}, Nothing} = nfmu.x0,
    tspan::Tuple{Float64, Float64} = nfmu.tspan;
    showProgress::Bool = false,
    progressDescr::String=DEFAULT_PROGRESS_DESCR,
    tolerance::Union{Real, Nothing} = nothing,
    parameters::Union{Dict{<:Any, <:Any}, Nothing} = nothing,
    p=nfmu.p,
    solver=nfmu.solver, 
    saveEventPositions::Bool=false,
    max_execution_duration::Real=-1.0,
    recordValues::fmi2ValueReferenceFormat=nfmu.recordValues,
    recordEigenvaluesSensitivity::Symbol=:none,
    recordEigenvalues::Bool=(recordEigenvaluesSensitivity != :none), 
    saveat=nfmu.saveat, # ToDo: Data type 
    sensealg=nfmu.fmu.executionConfig.sensealg, # ToDo: AbstractSensitivityAlgorithm
    writeSnapshot::Union{FMUSnapshot, Nothing}=nothing,
    readSnapshot::Union{FMUSnapshot, Nothing}=nothing,
    cleanSnapshots::Bool=true,
    solvekwargs...)

    if !isnothing(saveat)
        if saveat[1] != tspan[1] || saveat[end] != tspan[end] 
            logWarning(nfmu.fmu, "NeuralFMU changed time interval, start time is $(tspan[1]) and stop time is $(tspan[end]), but saveat from constructor gives $(saveat[1]) and $(saveat[end]).\nPlease provide correct `saveat` via keyword with matching start/stop time.", 1)
            saveat = collect(saveat)
            while saveat[1] < tspan[1]
                popfirst!(saveat)
            end
            while saveat[end] > tspan[end]
                pop!(saveat)
            end
        end
    end
    
    recordValues = prepareValueReference(nfmu.fmu, recordValues)

    saving = (length(recordValues) > 0)
    
    t_start = tspan[1]
    t_stop = tspan[end]

    nfmu.tspan = tspan
    nfmu.x0 = x_start
    nfmu.p = p 

    ignore_derivatives() do
        @debug "ME_NeuralFMU(showProgress=$(showProgress), tspan=$(tspan), x0=$(nfmu.x0))"

        nfmu.firstRun = true

        nfmu.tolerance = tolerance

        if isnothing(parameters)
            if !isnothing(nfmu.fmu.default_p_refs)
                nfmu.parameters = Dict(nfmu.fmu.default_p_refs .=> unsense(nfmu.fmu.default_p))
            end
        else
            nfmu.parameters = parameters
        end
    
    end

    callbacks = []

    c = getInstance(nfmu)

    @debug "ME_NeuralFMU: Starting callback..."
    c = startCallback(nothing, nfmu, c, t_start, writeSnapshot, readSnapshot)
    
    ignore_derivatives() do

        c.solution = FMUSolution(c)

        @debug "ME_NeuralFMU: Defining callbacks..."

        # custom callbacks
        for cb in nfmu.customCallbacksBefore
            push!(callbacks, cb)
        end

        nfmu.fmu.hasStateEvents = (c.fmu.modelDescription.numberOfEventIndicators > 0)
        nfmu.fmu.hasTimeEvents = (c.eventInfo.nextEventTimeDefined == fmi2True)

        # time event handling

        if nfmu.fmu.executionConfig.handleTimeEvents && nfmu.fmu.hasTimeEvents
            timeEventCb = IterativeCallback((integrator) -> time_choice(nfmu, c, integrator, t_start, t_stop),
            (integrator) -> affectFMU!(nfmu, c, integrator, 0), 
            Float64; 
            initial_affect=(c.eventInfo.nextEventTime == t_start), # already checked in the outer closure: c.eventInfo.nextEventTimeDefined == fmi2True
            save_positions=(saveEventPositions, saveEventPositions))

            push!(callbacks, timeEventCb)
        end

        # state event callback

        if c.fmu.hasStateEvents && c.fmu.executionConfig.handleStateEvents

            handleIndicators = nothing

            # if we want a specific subset
            if !isnothing(c.fmu.handleEventIndicators)
                handleIndicators = c.fmu.handleEventIndicators
            else # handle all
                handleIndicators = collect(UInt32(i) for i in 1:c.fmu.modelDescription.numberOfEventIndicators)
            end

            numEventInds = length(handleIndicators)

            if c.fmu.executionConfig.useVectorCallbacks

                eventCb = VectorContinuousCallback((out, x, t, integrator) -> condition!(nfmu, c, out, x, t, integrator, handleIndicators),
                                                (integrator, idx) -> affectFMU!(nfmu, c, integrator, idx),
                                                numEventInds;
                                                rootfind=RightRootFind,
                                                save_positions=(saveEventPositions, saveEventPositions),
                                                interp_points=c.fmu.executionConfig.rootSearchInterpolationPoints)
                push!(callbacks, eventCb)
            else

                for idx in 1:c.fmu.modelDescription.numberOfEventIndicators
                    eventCb = ContinuousCallback((x, t, integrator) -> conditionSingle(nfmu, c, idx, x, t, integrator),
                                                    (integrator) -> affectFMU!(nfmu, c, integrator, idx);
                                                    rootfind=RightRootFind,
                                                    save_positions=(saveEventPositions, saveEventPositions),
                                                    interp_points=c.fmu.executionConfig.rootSearchInterpolationPoints)
                    push!(callbacks, eventCb)
                end
            end
        end

        if max_execution_duration > 0.0
            terminateCb = ContinuousCallback((x, t, integrator) -> checkExecTime(integrator, nfmu, c, max_execution_duration),
                                                    (integrator) -> terminate!(integrator);
                                                    save_positions=(false, false))
            push!(callbacks, terminateCb)
            logInfo(nfmu.fmu, "Setting max execeution time to $(max_execution_duration)")
        end

        # custom callbacks
        for cb in nfmu.customCallbacksAfter
            push!(callbacks, cb)
        end

        if showProgress
            c.progressMeter = ProgressMeter.Progress(1000; desc=progressDescr, color=:blue, dt=1.0) 
            ProgressMeter.update!(c.progressMeter, 0) # show it!
        else
            c.progressMeter = nothing
        end

        # integrator step callback
        stepCb = FunctionCallingCallback((x, t, integrator) -> stepCompleted(nfmu, c, x, t, integrator, t_start, t_stop);
                                            func_everystep=true,
                                            func_start=true)
        push!(callbacks, stepCb)

        # [ToDo] Allow for AD-primitives for sensitivity analysis of recorded values
        if saving
            c.solution.values = SavedValues(Float64, Tuple{collect(Float64 for i in 1:length(recordValues))...})
            c.solution.valueReferences = recordValues

            if isnothing(saveat)
                savingCB = SavingCallback((x, t, integrator) -> saveValues(nfmu, c, recordValues, x, t, integrator),
                                c.solution.values)
            else
                savingCB = SavingCallback((x, t, integrator) -> saveValues(nfmu, c, recordValues, x, t, integrator),
                                c.solution.values,
                                saveat=saveat)
            end
            push!(callbacks, savingCB)
        end

        if recordEigenvalues

            @assert recordEigenvaluesSensitivity ∈ (:none, :ForwardDiff, :ReverseDiff, :Zygote) "Keyword `recordEigenvaluesSensitivity` must be one of (:none, :ForwardDiff, :ReverseDiff, :Zygote)"
            
            recordEigenvaluesType = nothing
            if recordEigenvaluesSensitivity == :ForwardDiff 
                recordEigenvaluesType = FMISensitivity.ForwardDiff.Dual 
            elseif recordEigenvaluesSensitivity == :ReverseDiff 
                recordEigenvaluesType = FMISensitivity.ReverseDiff.TrackedReal 
            elseif recordEigenvaluesSensitivity ∈ (:none, :Zygote)
                recordEigenvaluesType = fmi2Real
            end

            dtypes = collect(recordEigenvaluesType for _ in 1:2*length(c.fmu.modelDescription.stateValueReferences))
            c.solution.eigenvalues = SavedValues(recordEigenvaluesType, Tuple{dtypes...})
            
            savingCB = nothing
            if isnothing(saveat)
                savingCB = SavingCallback((u,t,integrator) -> saveEigenvalues(nfmu, c, u, t, integrator, recordEigenvaluesSensitivity), 
                                          c.solution.eigenvalues)
            else
                savingCB = SavingCallback((u,t,integrator) -> saveEigenvalues(nfmu, c, u, t, integrator, recordEigenvaluesSensitivity), 
                                        c.solution.eigenvalues, 
                                        saveat=saveat)
            end
            push!(callbacks, savingCB)
        end

    end # ignore_derivatives

    prob = nothing

    function fx_ip(dx, x, p, t)
        fx(nfmu, c, dx, x, p, t)
        return nothing
    end

    # function fx_op(x, p, t)
    #     return fx(nfmu, c, x, p, t)
    # end
    
    # function fx_jac(J, x, p, t)
    #     J[:] = ReverseDiff.jacobian(_x -> fx_op(_x, p, t), x)
    #     return nothing 
    # end

    # function jvp(Jv, v, x, p, t)
    #     n = length(x)
    #     J = similar(x, (n, n))
    #     fx_jac(J, x, p, t)
    #     Jv[:] = J * v 
    #     return nothing
    # end

    # function vjp(Jv, v, x, p, t)
    #     n = length(x)
    #     J = similar(x, (n, n))
    #     fx_jac(J, x, p, t)
    #     Jv[:] = v' * J
    #     return nothing
    # end

    ff = ODEFunction{true}(fx_ip) # ; jvp=jvp, vjp=vjp, jac=fx_jac) # tgrad=nothing
    prob = ODEProblem{true}(ff, nfmu.x0, nfmu.tspan, p)

    # [TODO] that (using ReverseDiffAdjoint) should work now with `autodiff=false` 
    if isnothing(sensealg)
        if isnothing(solver)

            logWarning(nfmu.fmu, "No solver keyword detected for NeuralFMU.\nContinuous adjoint method is applied, which requires solving backward in time.\nThis might be not supported by every FMU.", 1)
            sensealg = InterpolatingAdjoint(; autojacvec=ReverseDiffVJP(true), checkpointing=true)
        elseif isimplicit(solver)
            @assert !(alg_autodiff(solver) isa AutoForwardDiff) "Implicit solver using `autodiff=true` detected for NeuralFMU.\nThis is currently not supported, please use `autodiff=false` as solver keyword.\nExample: `Rosenbrock23(autodiff=false)` instead of `Rosenbrock23()`."

            logWarning(nfmu.fmu, "Implicit solver detected for NeuralFMU.\nContinuous adjoint method is applied, which requires solving backward in time.\nThis might be not supported by every FMU.", 1)
            sensealg = InterpolatingAdjoint(; autojacvec=ReverseDiffVJP(true), checkpointing=true)
        else
            sensealg = ReverseDiffAdjoint() 
        end
    end

    args = Vector{Any}()
    kwargs = Dict{Symbol, Any}(nfmu.solvekwargs..., solvekwargs...)

    if !isnothing(saveat)
        kwargs[:saveat] = saveat
    end

    ignore_derivatives() do
        if !isnothing(solver)
            push!(args, solver)
        end 
    end

    #kwargs[:callback]=CallbackSet(callbacks...)
    #kwargs[:sensealg]=sensealg
    #kwargs[:u0] = nfmu.x0 # this is because of `IntervalNonlinearProblem has no field u0`
 
    @debug "ME_NeuralFMU: Start solving ..."
     
    c.solution.states = solve(prob, args...; callback=CallbackSet(callbacks...), sensealg=sensealg, u0=nfmu.x0, kwargs...) 

    @debug "ME_NeuralFMU: ... finished solving!"

    ignore_derivatives() do

        @assert !isnothing(c.solution.states) "Solving NeuralODE returned `nothing`!"
 
        # ReverseDiff returns an array instead of an ODESolution, this needs to be corrected
        # [TODO] doesn`t Array cover the TrackedArray case?
        if isa(c.solution.states, TrackedArray) || isa(c.solution.states, Array) 

            @assert !isnothing(saveat) "Keyword `saveat` is nothing, please provide the keyword when using ReverseDiff."
           
            t = collect(saveat)
            while t[1] < tspan[1]
                popfirst!(t)
            end
            while t[end] > tspan[end]
                pop!(t)
            end
            u = c.solution.states
            c.solution.success = (size(u) == (length(nfmu.x0), length(t))) 

            if size(u)[2] > 0 # at least some recorded points
                c.solution.states = build_solution(prob, solver, t, collect(u[:,i] for i in 1:size(u)[2]))
            end
        else
            c.solution.success = (c.solution.states.retcode == ReturnCode.Success)
        end
        
    end # ignore_derivatives

    @debug "ME_NeuralFMU: Stopping callback..."

    # stopCB 
    stopCallback(nfmu, c, t_stop)

    # cleanup snapshots to release memory
    if cleanSnapshots
        for snapshot in c.solution.snapshots 
            FMIBase.freeSnapshot!(snapshot)
        end
        c.solution.snapshots = Vector{FMUSnapshot}(undef, 0)
    end

    return c.solution
end
function (nfmu::ME_NeuralFMU)(x0::Union{Array{<:Real}, Nothing},
                              t::Real;
                              p=nothing)

    c = nothing

    return fx(nfmu, c, x0, p, t)
end

"""

    ToDo: Docstring for Arguments, Keyword arguments, ...

Evaluates the CS_NeuralFMU in the timespan given during construction or in a custum timespan from `t_start` to `t_stop` with a given time step size `t_step`.

Via optional argument `reset`, the FMU is reset every time evaluation is started (default=`true`).
"""
function (nfmu::CS_NeuralFMU{F, C})(inputFct,
    t_step::Real, 
    tspan::Tuple{Float64, Float64} = nfmu.tspan; 
    p=nfmu.p,
    tolerance::Union{Real, Nothing} = nothing,
    parameters::Union{Dict{<:Any, <:Any}, Nothing} = nothing) where {F, C}

    t_start, t_stop = tspan

    c = (hasCurrentInstance(nfmu.fmu) ? getCurrentInstance(nfmu.fmu) : nothing)
    c, _ = prepareSolveFMU(nfmu.fmu, c, fmi2TypeCoSimulation; parameters=parameters, t_start=t_start, t_stop=t_stop, tolerance=tolerance, cleanup=true)
    
    ts = collect(t_start:t_step:t_stop)
   
    model_input = inputFct.(ts)

    firstStep = true
    function simStep(input)
        y = nothing 

        if !firstStep
            ignore_derivatives() do
                fmi2DoStep(c, t_step)
            end
        else
            firstStep = false
        end

        if p == nothing # structured, implicite parameters
            y = nfmu.model(input)
        else # flattened, explicite parameters
            @assert !isnothing(nfmu.re) "Using explicite parameters without destructing the model."
            y = nfmu.re(p)(input)
        end
        
        return y
    end

    valueStack = simStep.(model_input)

    ignore_derivatives() do
        c.solution.success = true
    end

    c.solution.values = SavedValues{typeof(ts[1]), typeof(valueStack[1])}(ts, valueStack)

    # [ToDo] check if this is still the case for current releases of related libraries
    # this is not possible in CS (pullbacks are sometimes called after the finished simulation), clean-up happens at the next call
    # c = finishSolveFMU(nfmu.fmu, c, freeInstance, terminate)

    return c.solution
end

function (nfmu::CS_NeuralFMU{Vector{F}, Vector{C}})(inputFct,
                                         t_step::Real, 
                                         tspan::Tuple{Float64, Float64} = nfmu.tspan; 
                                         p=nothing,
                                         tolerance::Union{Real, Nothing} = nothing,
                                         parameters::Union{Vector{Union{Dict{<:Any, <:Any}, Nothing}}, Nothing} = nothing) where {F, C}

    t_start, t_stop = tspan
    numFMU = length(nfmu.fmu)

    cs = nothing
    ignore_derivatives() do
        cs = Vector{Union{FMU2Component, Nothing}}(undef, numFMU)
        for i in 1:numFMU
            cs[i] = (hasCurrentInstance(nfmu.fmu[i]) ? getCurrentInstance(nfmu.fmu[i]) : nothing)
        end
    end
    for i in 1:numFMU
        cs[i], _ = prepareSolveFMU(nfmu.fmu[i], cs[i], fmi2TypeCoSimulation; parameters=parameters, t_start=t_start, t_stop=t_stop, tolerance=tolerance, cleanup=true)
    end
    
    solution = FMUSolution(nothing)

    ts = collect(t_start:t_step:t_stop)
    model_input = inputFct.(ts)

    firstStep = true
    function simStep(input)
        y = nothing

        if !firstStep
            ignore_derivatives() do
                for c in cs
                    fmi2DoStep(c, t_step)
                end
            end
        else
            firstStep = false
        end

        if p == nothing # structured, implicite parameters
            y = nfmu.model(input)
        else # flattened, explicite parameters
            @assert nfmu.re != nothing "Using explicite parameters without destructing the model."
           
            if length(p) == 1
                y = nfmu.re(p[1])(input)
            else
                y = nfmu.re(p)(input)
            end
        end

        return y
    end

    valueStack = simStep.(model_input)

    ignore_derivatives() do
        solution.success = true # ToDo: Check successful simulation!
    end 
    
    solution.values = SavedValues{typeof(ts[1]), typeof(valueStack[1])}(ts, valueStack)
    
    # [ToDo] check if this is still the case for current releases of related libraries
    # this is not possible in CS (pullbacks are sometimes called after the finished simulation), clean-up happens at the next call
    # cs = finishSolveFMU(nfmu.fmu, cs, freeInstance, terminate)

    return solution
end

# adapting the Flux functions
function Flux.params(nfmu::ME_NeuralFMU; destructure::Bool=false)
    if destructure 
        nfmu.p, nfmu.re = Flux.destructure(nfmu.model)
    end

    ps = Flux.params(nfmu.p)

    if issense(ps)
        @warn "Parameters include AD-primitives, this indicates that something did go wrong in before."
    end

    return ps
end

function Flux.params(nfmu::CS_NeuralFMU; destructure::Bool=false) # true)
    if destructure 
        nfmu.p, nfmu.re = Flux.destructure(nfmu.model)
        
    # else
    #     return Flux.params(nfmu.model)
    end

    ps = Flux.params(nfmu.p)

    if issense(ps)
        @warn "Parameters include AD-primitives, this indicates that something did go wrong in before."
    end

    return ps
end

function computeGradient!(jac, loss, params, gradient::Symbol, chunk_size::Union{Symbol, Int}, multiObjective::Bool)

    if gradient == :ForwardDiff

        if chunk_size == :auto_forwarddiff
            
            if multiObjective
                conf = ForwardDiff.JacobianConfig(loss, params)
                ForwardDiff.jacobian!(jac, loss, params, conf)
            else
                conf = ForwardDiff.GradientConfig(loss, params)
                ForwardDiff.gradient!(jac, loss, params, conf)
            end

        elseif chunk_size == :auto_fmiflux

            chunk_size = DEFAULT_CHUNK_SIZE
            
            if multiObjective
                conf = ForwardDiff.JacobianConfig(loss, params, ForwardDiff.Chunk{min(chunk_size, length(params))}());
                ForwardDiff.jacobian!(jac, loss, params, conf)
            else
                conf = ForwardDiff.GradientConfig(loss, params, ForwardDiff.Chunk{min(chunk_size, length(params))}());
                ForwardDiff.gradient!(jac, loss, params, conf)
            end
        else

            if multiObjective
                conf = ForwardDiff.JacobianConfig(loss, params, ForwardDiff.Chunk{min(chunk_size, length(params))}());
                ForwardDiff.jacobian!(jac, loss, params, conf)
            else
                conf = ForwardDiff.GradientConfig(loss, params, ForwardDiff.Chunk{min(chunk_size, length(params))}());
                ForwardDiff.gradient!(jac, loss, params, conf)
            end
        end

    elseif gradient == :Zygote 

        if multiObjective
            jac[:] = Zygote.jacobian(loss, params)[1]
        else
            jac[:] = Zygote.gradient(loss, params)[1]
        end

    elseif gradient == :ReverseDiff 

        if multiObjective
            ReverseDiff.jacobian!(jac, loss, params)
        else
            ReverseDiff.gradient!(jac, loss, params)
        end
    elseif gradient == :FiniteDiff 

        if multiObjective
            FiniteDiff.finite_difference_jacobian!(jac, loss, params)
        else
            FiniteDiff.finite_difference_gradient!(jac, loss, params)
        end
    else
        @assert false "Unknown `gradient=$(gradient)`, supported are `:ForwardDiff`, `:Zygote`, `:FiniteDiff` and `:ReverseDiff`."
    end

    ### check gradient is valid 

    # [Todo] Better!
    grads = nothing
    if multiObjective
        grads = collect(jac[i,:] for i in 1:size(jac)[1])
    else
        grads = [jac]
    end

    all_zero = any(collect(all(iszero.(grad)) for grad in grads))
    has_nan = any(collect(any(isnan.(grad)) for grad in grads))
    has_nothing = any(collect(any(isnothing.(grad)) for grad in grads)) || any(isnothing.(grads))

    @assert !all_zero "Determined gradient containes only zeros.\nThis might be because the loss function is:\n(a) not sensitive regarding the model parameters or\n(b) sensitivities regarding the model parameters are not traceable via AD."

    if gradient != :ForwardDiff && (has_nan || has_nothing)
        @warn "Gradient determination with $(gradient) failed, because gradient contains `NaNs` and/or `nothing`.\nThis might be because the FMU is throwing redundant events, which is currently not supported.\nTrying ForwardDiff as back-up.\nIf this message gets printed (almost) every step, consider using keyword `gradient=:ForwardDiff` to fix ForwardDiff as sensitivity system."
        gradient = :ForwardDiff
        computeGradient!(jac, loss, params, gradient, chunk_size, multiObjective)

        if multiObjective
            grads = collect(jac[i,:] for i in 1:size(jac)[1])
        else
            grads = [jac]
        end
    end

    has_nan = any(collect(any(isnan.(grad)) for grad in grads))
    has_nothing = any(collect(any(isnothing.(grad)) for grad in grads)) || any(isnothing.(grads))
        
    @assert !has_nan "Gradient determination with $(gradient) failed, because gradient contains `NaNs`.\nNo back-up options available."
    @assert !has_nothing "Gradient determination with $(gradient) failed, because gradient contains `nothing`.\nNo back-up options available."

    return nothing
end

lk_TrainApply = ReentrantLock()
function trainStep(loss, params, gradient, chunk_size, optim::FMIFlux.AbstractOptimiser, printStep, proceed_on_assert, cb, multiObjective)

    global lk_TrainApply
    
    try
                
        for j in 1:length(params)

            step = FMIFlux.apply!(optim, params[j])

            lock(lk_TrainApply) do
                
                params[j] .-= step

                if printStep
                    @info "Step: min(abs()) = $(min(abs.(step)...))   max(abs()) = $(max(abs.(step)...))"
                end
                
            end

        end    

    catch e

        if proceed_on_assert
            msg = "$(e)"
            msg = length(msg) > 4096 ? first(msg, 4096) * "..." : msg
            @error "Training asserted, but continuing: $(msg)"
        else
            throw(e)
        end
    end

    if cb != nothing 
        if isa(cb, AbstractArray)
            for _cb in cb 
                _cb()
            end
        else
            cb()
        end
    end

end

"""

    train!(loss, neuralFMU::Union{ME_NeuralFMU, CS_NeuralFMU}, data, optim; gradient::Symbol=:ReverseDiff, kwargs...)

A function analogous to Flux.train! but with additional features and explicit parameters (faster).

# Arguments
- `loss` a loss function in the format `loss(p)`
- `neuralFMU` a object holding the neuralFMU with its parameters
- `data` the training data (or often an iterator)
- `optim` the optimizer used for training 

# Keywords 
- `gradient` a symbol determining the AD-library for gradient computation, available are `:ForwardDiff`, `:Zygote` and :ReverseDiff (default)
- `cb` a custom callback function that is called after every training step (default `nothing`)
- `chunk_size` the chunk size for AD using ForwardDiff (ignored for other AD-methods) (default `:auto_fmiflux`)
- `printStep` a boolean determining wheater the gradient min/max is printed after every step (for gradient debugging) (default `false`)
- `proceed_on_assert` a boolean that determins wheater to throw an ecxeption on error or proceed training and just print the error (default `false`)
- `multiThreading`: a boolean that determins if multiple gradients are generated in parallel (default `false`)
- `multiObjective`: set this if the loss function returns multiple values (multi objective optimization), currently gradients are fired to the optimizer one after another (default `false`)
"""
function train!(loss, neuralFMU::Union{ME_NeuralFMU, CS_NeuralFMU}, data, optim; gradient::Symbol=:ReverseDiff, kwargs...)
    params = Flux.params(neuralFMU) 

    snapshots = neuralFMU.snapshots

    # [Note] :ReverseDiff, :Zygote need it for state change sampling and the rrule
    #        :ForwardDiff needs it for state change sampling
    neuralFMU.snapshots = true
   
    _train!(loss, params, data, optim; gradient=gradient, kwargs...)

    neuralFMU.snapshots = snapshots
    neuralFMU.p = unsense(neuralFMU.p)

    return nothing
end

# Dispatch for FMIFlux.jl [FMIFlux.AbstractOptimiser]
function _train!(loss, 
    params::Union{Flux.Params, Zygote.Params, AbstractVector{<:AbstractVector{<:Real}}}, 
    data, 
    optim::FMIFlux.AbstractOptimiser; 
    gradient::Symbol=:ReverseDiff, 
    cb=nothing, chunk_size::Union{Integer, Symbol}=:auto_fmiflux, 
    printStep::Bool=false, 
    proceed_on_assert::Bool=false, 
    multiThreading::Bool=false, 
    multiObjective::Bool=false)

    if length(params) <= 0 || length(params[1]) <= 0 
        @warn "train!(...): Empty parameter array, training on an empty parameter array doesn't make sense."
        return 
    end

    if multiThreading && Threads.nthreads() == 1 
        @warn "train!(...): Multi-threading is set via flag `multiThreading=true`, but this Julia process does not have multiple threads. This will not result in a speed-up. Please spawn Julia in multi-thread mode to speed-up training."
    end

    _trainStep = (i,) -> trainStep(loss, params, gradient, chunk_size, optim, printStep, proceed_on_assert, cb, multiObjective)

    if multiThreading
        ThreadPools.qforeach(_trainStep, 1:length(data))
    else
        foreach(_trainStep, 1:length(data))
    end

end

# Dispatch for Flux.jl [Flux.Optimise.AbstractOptimiser]
function _train!(loss, params::Union{Flux.Params, Zygote.Params, AbstractVector{<:AbstractVector{<:Real}}}, data, optim::Flux.Optimise.AbstractOptimiser; gradient::Symbol=:ReverseDiff, chunk_size::Union{Integer, Symbol}=:auto_fmiflux, multiObjective::Bool=false, kwargs...) 
    
    grad_buffer = nothing

    if multiObjective
        dim = loss(params[1])

        grad_buffer = zeros(Float64, length(params[1]), length(dim))
    else
        grad_buffer = zeros(Float64, length(params[1])) 
    end

    grad_fun! = (G, p) -> computeGradient!(G, loss, p, gradient, chunk_size, multiObjective)
    _optim = FluxOptimiserWrapper(optim, grad_fun!, grad_buffer)
    _train!(loss, params, data, _optim; gradient=gradient, chunk_size=chunk_size, multiObjective=multiObjective, kwargs...)
end

# Dispatch for Optim.jl [Optim.AbstractOptimizer]
function _train!(loss, params::Union{Flux.Params, Zygote.Params, AbstractVector{<:AbstractVector{<:Real}}}, data, optim::Optim.AbstractOptimizer; gradient::Symbol=:ReverseDiff, chunk_size::Union{Integer, Symbol}=:auto_fmiflux, multiObjective::Bool=false, kwargs...) 
    if length(params) <= 0 || length(params[1]) <= 0 
        @warn "train!(...): Empty parameter array, training on an empty parameter array doesn't make sense."
        return 
    end
    
    grad_fun! = (G, p) -> computeGradient!(G, loss, p, gradient, chunk_size, multiObjective)
    _optim = OptimOptimiserWrapper(optim, grad_fun!, loss, params[1])
    _train!(loss, params, data, _optim; gradient=gradient, chunk_size=chunk_size, multiObjective=multiObjective, kwargs...)
end
