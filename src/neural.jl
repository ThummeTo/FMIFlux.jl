#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMIImport.FMICore: assert_integrator_valid, isdual, istracked, undual, unsense, untrack
import FMIImport: finishSolveFMU, handleEvents, prepareSolveFMU
import Optim
import ProgressMeter
import FMISensitivity.SciMLSensitivity.SciMLBase: CallbackSet, ContinuousCallback, ODESolution, ReturnCode, RightRootFind,
    VectorContinuousCallback, set_u!, terminate!, u_modified!, build_solution
using FMISensitivity.ReverseDiff: TrackedArray
import FMISensitivity.SciMLSensitivity: InterpolatingAdjoint, ReverseDiffVJP
import ThreadPools

using DifferentialEquations.DiffEqCallbacks
using DifferentialEquations: ODEFunction, ODEProblem, solve
using FMIImport: FMU2Component, FMU2Event, FMU2Solution, fmi2ComponentState,
    fmi2ComponentStateContinuousTimeMode, fmi2ComponentStateError,
    fmi2ComponentStateEventMode, fmi2ComponentStateFatal,
    fmi2ComponentStateInitializationMode, fmi2ComponentStateInstantiated,
    fmi2ComponentStateTerminated, fmi2StatusOK, fmi2Type, fmi2TypeCoSimulation,
    fmi2TypeModelExchange, logError, logInfo, logWarning 
using FMISensitivity.SciMLSensitivity:
    ForwardDiffSensitivity, InterpolatingAdjoint, ReverseDiffVJP, ZygoteVJP
import DifferentiableEigen


import FMIImport.FMICore: EMPTY_fmi2Real, EMPTY_fmi2ValueReference

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
    kwargs

    # re_model 
    # re_p

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
    setup::Union{Bool, Nothing}
    reset::Union{Bool, Nothing}
    instantiate::Union{Bool, Nothing}
    freeInstance::Union{Bool, Nothing}
    terminate::Union{Bool, Nothing}

    modifiedState::Bool

    execution_start::Real

    function ME_NeuralFMU{M, R}(model::M, p::AbstractArray{<:Real}, re::R) where {M, R}
        inst = new()
        inst.model = model 
        inst.p = p 
        inst.re = re 
        inst.x0 = nothing
        inst.saveat = nothing

        # inst.re_model = nothing
        # inst.re_p = nothing

        inst.modifiedState = false

        # inst.startState = nothing 
        # inst.stopState = nothing
        # inst.startEventInfo = nothing 
        # inst.stopEventInfo = nothing

        inst.customCallbacksBefore = []
        inst.customCallbacksAfter = []

        inst.execution_start = 0.0
       
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

    function CS_NeuralFMU{F, C}() where {F, C}
        inst = new{F, C}()

        inst.re = nothing
        inst.p = nothing

        return inst
    end
end

function evaluateModel(nfmu::ME_NeuralFMU, c::FMU2Component, x; p=nfmu.p)
    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    # [ToDo]: Cache the restructure if possible
    # if isnothing(nfmu.re_model) || p != nfmu.re_p
    #     nfmu.re_p = p # fast_copy!(nfmu, :re_p, p)
    #     nfmu.re_model = nfmu.re(p)
    # end
    # return nfmu.re_model(x)

    nfmu.p = p 
    return nfmu.re(p)(x)
end

function evaluateModel(nfmu::ME_NeuralFMU, c::FMU2Component, dx, x; p=nfmu.p)
    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    # [ToDo]: Cache the restructure if possible
    # if isnothing(nfmu.re_model) || p != nfmu.re_p
    #     nfmu.re_p = p # fast_copy!(nfmu, :re_p, p)
    #     nfmu.re_model = nfmu.re(p)
    # end
    # dx[:] = nfmu.re_model(x)

    nfmu.p = p 
    dx[:] = nfmu.re(p)(x)

    return nothing
end

##### EVENT HANDLING START

function startCallback(integrator, nfmu::ME_NeuralFMU, c::Union{FMU2Component, Nothing}, t)

    ignore_derivatives() do

        nfmu.execution_start = time()

        t = unsense(t)

        @assert t == nfmu.tspan[1] "startCallback(...): Called for non-start-point t=$(t)"
        
        c, x0 = prepareSolveFMU(nfmu.fmu, c, fmi2TypeModelExchange, nfmu.instantiate, nfmu.freeInstance, nfmu.terminate, nfmu.reset, nfmu.setup, nfmu.parameters, nfmu.tspan[1], nfmu.tspan[end], nfmu.tolerance; x0=nfmu.x0, handleEvents=FMIFlux.handleEvents, cleanup=true)
        
        if c.eventInfo.nextEventTime == t && c.eventInfo.nextEventTimeDefined == fmi2True
            @debug "Initial time event detected!"
        else
            @debug "No initial time events ..."
        end

    end

    return c
end

function stopCallback(nfmu::ME_NeuralFMU, c::FMU2Component, t)

    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    ignore_derivatives() do
        
        t = unsense(t)

        @assert t == nfmu.tspan[end] "stopCallback(...): Called for non-start-point t=$(t)"
    end

    return c
end

# Read next time event from fmu and provide it to the integrator 
function time_choice(nfmu::ME_NeuralFMU, c::FMU2Component, integrator, tStart, tStop)

    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @assert c.fmu.executionConfig.handleTimeEvents "time_choice(...) was called, but execution config disables time events.\nPlease open a issue."
    @debug assert_integrator_valid(integrator)

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
        logWarning(nfmu.fmu, "There is currently an issue with the condition buffer pre-allocation, the buffer can't be overwritten by the generated rrule.")
        out[:] = zeros(fmi2Real, length(out))
    end
    
    invoke(condition!, Tuple{ME_NeuralFMU, FMU2Component, Any,  Any, Any, Any, Any}, nfmu, c, out, x, t, integrator, handleEventIndicators)
    
    return nothing
end

function condition!(nfmu::ME_NeuralFMU, c::FMU2Component, out, x, t, integrator, handleEventIndicators) 

    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @assert c.state == fmi2ComponentStateContinuousTimeMode "condition!(...):\n" * FMICore.ERR_MSG_CONT_TIME_MODE

    # [ToDo] Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN.
    #        Basically only the layers from very top to FMU need to be evaluated here.
    c.default_t = t
    c.default_ec = out
    c.default_ec_idcs = handleEventIndicators
    evaluateModel(nfmu, c, x)

    # [TODO] for generic applications, reset to previous values
    c.default_t = -1.0
    c.default_ec = EMPTY_fmi2Real
    c.default_ec_idcs = EMPTY_fmi2ValueReference

    # write back to condition buffer
    out[:] = c.eval_output.ec # [ToDo] This seems not to be necessary, because of `c.default_ec = out`
    
    c.solution.evals_condition += 1

    @debug "condition!(...) -> [typeof=$(typeof(out))]\n$(unsense(out))"

    return nothing
end

global lastIndicator = nothing
global lastIndicatorX = nothing 
global lastIndicatorT = nothing
function conditionSingle(nfmu::ME_NeuralFMU, c::FMU2Component, index, x, t, integrator) 

    @assert c.state == fmi2ComponentStateContinuousTimeMode "condition(...):\n" * FMICore.ERR_MSG_CONT_TIME_MODE
    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

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

# [ToDo] Check, that the new determined state is the right root of the event instant!
function f_optim(x, nfmu::ME_NeuralFMU, c::FMU2Component, right_x_fmu) # , idx, direction::Real)
    # propagete the new state-guess `x` through the NeuralFMU
    evaluateModel(nfmu, c, x)
    #indicators = fmi2GetEventIndicators(c)
    return Flux.Losses.mae(right_x_fmu, fmi2GetContinuousStates(c)) # - min(-direction*indicators[idx], 0.0)
end

# Handles the upcoming event
function affectFMU!(nfmu::ME_NeuralFMU, c::FMU2Component, integrator, idx)

    @debug "affectFMU!"
    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    assert_integrator_valid(integrator)

    @assert c.state == fmi2ComponentStateContinuousTimeMode "affectFMU!(...):\n" * FMICore.ERR_MSG_CONT_TIME_MODE

    # [NOTE] Here unsensing is OK, because we just want to reset the FMU to the correct state!
    #        The values come directly from the integrator and are NOT function arguments!
    t = integrator.t # unsense(integrator.t)
    x = integrator.u # unsense(integrator.u)
    
    if c.x != x
        # capture status of `force`    
        mode = c.force
        c.force = true

        # there are fx-evaluations before the event is handled, reset the FMU state to the current integrator step
        c.default_t = t
        evaluateModel(nfmu, c, x) # evaluate NeuralFMU (set new states)
        # [NOTE] No need to reset time here, because we did pass a event instance! 
        #        c.default_t = -1.0

        c.force = mode
    end

    fmi2EnterEventMode(c)
    handleEvents(c)

    left_x = nothing
    right_x = nothing

    if c.eventInfo.valuesOfContinuousStatesChanged == fmi2True

        left_x = unsense(x)
        right_x_fmu = fmi2GetContinuousStates(c) # the new FMU state after handled events

        ignore_derivatives() do 
            @debug "affectFMU!(_, _, $idx): NeuralFMU state event from $(left_x) (fmu: $(left_x_fmu)). Indicator [$idx]: $(indicators[idx]). Optimizing new state ..."
        end

        # [ToDo] use gradient-based optimization here?
        # if there is an ANN above the FMU, propaget FMU state through top ANN:
        if nfmu.modifiedState 
            result = Optim.optimize(x_seek -> f_optim(x_seek, nfmu, c, right_x_fmu), left_x, Optim.NelderMead())
            right_x = Optim.minimizer(result)
        else # if there is no ANN above, then:
            right_x = right_x_fmu
        end

        # [ToDo] This should only be done in the frule/rrule, the actual affect should do a hard "set state"
        for i in 1:length(left_x)
            if left_x[i] != 0.0 
                scale = right_x[i] ./ left_x[i]
                integrator.u[i] *= scale
            else # integrator state zero can't be scaled, need to add (but no sensitivities in this case!)
                shift = right_x[i] - left_x[i] 
                integrator.u[i] += shift
                logWarning(c.fmu, "Probably wrong sensitivities for ∂x^+ / ∂x^-\nCan't scale from zero state (state #$(i)=0.0)")
            end
        end
       
        u_modified!(integrator, true)
    else

        ignore_derivatives() do 
            @debug "affectFMU!(_, _, $idx): NeuralFMU event without state change at $t. Indicator [$idx]."
        end

        u_modified!(integrator, false)
    end

    if c.eventInfo.nominalsOfContinuousStatesChanged == fmi2True
        # ToDo: Do something with that information, e.g. use for FiniteDiff sampling step size determination
        x_nom = fmi2GetNominalsOfContinuousStates(c)
    end

    ignore_derivatives() do
        if idx != -1
            e = FMU2Event(unsense(t), UInt64(idx), unsense(left_x), unsense(right_x))
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
            logError(c.fmu, "Event chattering detected $(round(Integer, ratio)) events/s, aborting at t=$(t) (rel. t=$(pt)) at event $(ne):")
            for i in 0:c.fmu.modelDescription.numberOfEventIndicators
                num = 0
                for e in c.solution.events
                    if e.indicator == i
                        num += 1 
                    end 
                end
                if num > 0
                    logError(c.fmu, "\tEvent indicator #$(i) triggered $(num) ($(round(num/1000.0*100.0; digits=1))%)")
                end
            end

            terminate!(integrator)
        end
    end

    c.solution.evals_affect += 1

    assert_integrator_valid(integrator)
end

# Does one step in the simulation.
function stepCompleted(nfmu::ME_NeuralFMU, c::FMU2Component, x, t, integrator, tStart, tStop)

    assert_integrator_valid(integrator)

    c.solution.evals_stepcompleted += 1

    if !isnothing(c.progressMeter)
        ProgressMeter.update!(c.progressMeter, floor(Integer, 1000.0*(t-tStart)/(tStop-tStart)) )
    end

    if c != nothing
        (status, enterEventMode, terminateSimulation) = fmi2CompletedIntegratorStep(c, fmi2True)

        if terminateSimulation == fmi2True
            logError(c.fmu, "stepCompleted(...): FMU requested termination!")
        end

        if enterEventMode == fmi2True
            affectFMU!(nfmu, c, integrator, -1)
        end

        @debug "Step completed at $(ForwardDiff.value(t)) with $(collect(ForwardDiff.value(xs) for xs in x))"
    end

    assert_integrator_valid(integrator)
end

# [ToDo] (1) This must be in-place 
#        (2) getReal must be replaced with the inplace getter within c(...)
#        (3) remove unsense to determine save value sensitivities
# save FMU values 
function saveValues(nfmu::ME_NeuralFMU, c::FMU2Component, recordValues, _x, t, integrator)

    t = unsense(t) 
    x = unsense(_x)

    c.solution.evals_savevalues += 1

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    c.default_t = t
    evaluateModel(nfmu, c, x) # evaluate NeuralFMU (set new states)
   
    # Todo set inputs
    return (fmi2GetReal(c, recordValues)...,)
end

function saveEigenvalues(nfmu::ME_NeuralFMU, c::FMU2Component, _x, _t, integrator, sensitivity::Symbol)

    @assert c.state == fmi2ComponentStateContinuousTimeMode "saveEigenvalues(...):\n" * FMICore.ERR_MSG_CONT_TIME_MODE

    c.solution.evals_saveeigenvalues += 1

    c.default_t = t

    A = nothing
    if sensitivity == :ForwardDiff
        A = ForwardDiff.jacobian(x -> evaluateModel(nfmu, c, x), _x) # TODO: chunk_size!
    elseif sensitivity == :ReverseDiff 
        A = ReverseDiff.jacobian(x -> evaluateModel(nfmu, c, x), _x)
    elseif sensitivity == :Zygote 
        A = Zygote.jacobian(x -> evaluateModel(nfmu, c, x), _x)[1]
    elseif sensitivity == :none
        A = ForwardDiff.jacobian(x -> evaluateModel(nfmu, c, x), unsense(_x))
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

    ignore_derivatives() do
        c.default_t = t
    end 

    ############

    evaluateModel(nfmu, c, dx, x; p=p)

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

        c.default_t = t
    end

    return evaluateModel(nfmu, c, x; p=p)
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
                      kwargs...)

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
    nfmu.kwargs = kwargs
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

function getComponent(nfmu::NeuralFMU)
    return hasCurrentComponent(nfmu.fmu) ? getCurrentComponent(nfmu.fmu) : nothing
end

"""
    
    TODO: Signature, Arguments and Keyword-Arguments descriptions.

Evaluates the ME_NeuralFMU in the timespan given during construction or in a custom timespan from `t_start` to `t_stop` for a given start state `x_start`.

# Keyword arguments
    - `reset`, the FMU is reset every time evaluation is started (default=`true`).
    - `setup`, the FMU is set up every time evaluation is started (default=`true`).
"""
function (nfmu::ME_NeuralFMU)(x_start::Union{Array{<:Real}, Nothing} = nfmu.x0,
    tspan::Tuple{Float64, Float64} = nfmu.tspan;
    showProgress::Bool = false,
    progressDescr::String=DEFAULT_PROGRESS_DESCR,
    tolerance::Union{Real, Nothing} = nothing,
    parameters::Union{Dict{<:Any, <:Any}, Nothing} = nothing,
    setup::Union{Bool, Nothing} = nothing,
    reset::Union{Bool, Nothing} = nothing,
    instantiate::Union{Bool, Nothing} = nothing,
    freeInstance::Union{Bool, Nothing} = nothing,
    terminate::Union{Bool, Nothing} = nothing,
    p=nfmu.p,
    solver=nfmu.solver, 
    saveEventPositions::Bool=false,
    max_execution_duration::Real=-1.0,
    recordValues::fmi2ValueReferenceFormat=nfmu.recordValues,
    recordEigenvaluesSensitivity::Symbol=:none,
    recordEigenvalues::Bool=(recordEigenvaluesSensitivity != :none), 
    saveat=nfmu.saveat, # ToDo: Data type 
    sensealg=nfmu.fmu.executionConfig.sensealg, # ToDo: AbstractSensitivityAlgorithm
    kwargs...)

    # this shouldnt be forced
    # if !isnothing(saveat)
    #     if saveat[1] != tspan[1] 
    #         @warn "NeuralFMU changed time interval, start time is $(tspan[1]), but saveat from constructor gives $(saveat[1]). Please provide correct `saveat` via keyword with matching start/stop time."
    #     end
    #     if saveat[end] != tspan[end] 
    #         @warn "NeuralFMU changed time interval, stop time is $(tspan[end]), but saveat from constructor gives $(saveat[end]). Please provide correct `saveat` via keyword with matching start/stop time."
    #     end
    # end
    
    recordValues = prepareValueReference(nfmu.fmu, recordValues)

    saving = (length(recordValues) > 0)
    
    t_start = tspan[1]
    t_stop = tspan[end]

    nfmu.tspan = tspan
    nfmu.x0 = x_start
    nfmu.p = p 

    ignore_derivatives() do
        @debug "ME_NeuralFMU..."

        nfmu.firstRun = true

        nfmu.tolerance = tolerance

        if isnothing(parameters)
            if !isnothing(nfmu.fmu.default_p_refs)
                nfmu.parameters = Dict(nfmu.fmu.default_p_refs .=> unsense(nfmu.fmu.default_p))
            end
        else
            nfmu.parameters = parameters
        end
        nfmu.setup = setup
        nfmu.reset = reset
        nfmu.instantiate = instantiate
        nfmu.freeInstance = freeInstance
        nfmu.terminate = terminate
    end

    callbacks = []

    c = getComponent(nfmu)
    c = startCallback(nothing, nfmu, c, t_start)
    
    ignore_derivatives() do
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

            numEvents = length(handleIndicators)

            if c.fmu.executionConfig.useVectorCallbacks

                eventCb = VectorContinuousCallback((out, x, t, integrator) -> condition!(nfmu, c, out, x, t, integrator, handleIndicators),
                                                (integrator, idx) -> affectFMU!(nfmu, c, integrator, idx),
                                                numEvents;
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
                recordEigenvaluesType = FMIImport.ForwardDiff.Dual 
            elseif recordEigenvaluesSensitivity == :ReverseDiff 
                recordEigenvaluesType = FMIImport.ReverseDiff.TrackedReal 
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

    ff = ODEFunction{true}((dx, x, p, t) -> fx(nfmu, c, dx, x, p, t)) # tgrad=nothing
    prob = ODEProblem{true}(ff, nfmu.x0, nfmu.tspan, p)

    if isnothing(sensealg)
        sensealg = ReverseDiffAdjoint() 
    end

    args = Vector{Any}()
    kwargs = Dict{Symbol, Any}(kwargs...)

    if !isnothing(saveat)
        kwargs[:saveat] = saveat
    end

    if !isnothing(solver)
        push!(args, solver)
    end 
        
    c.solution.states = solve(prob, args...; sensealg=sensealg, callback=CallbackSet(callbacks...), nfmu.kwargs..., kwargs...) 

    ignore_derivatives() do

        @assert !isnothing(c.solution.states) "Solving NeuralODE returned `nothing`!"
 
        # ReverseDiff returns an array instead of an ODESolution, this needs to be corrected
        if isa(c.solution.states, TrackedArray)

            @assert !isnothing(saveat) "Keyword `saveat` is nothing, please provide the keyword."
           
            t = collect(saveat)
            u = c.solution.states
            c.solution.success = (size(u) == (length(nfmu.x0), length(t))) 

            if c.solution.success
                c.solution.states = build_solution(prob, nfmu.solver, t, collect(u[:,i] for i in 1:size(u)[2]))
            end
        else
            c.solution.success = (c.solution.states.retcode == ReturnCode.Success)
        end
        
    end # ignore_derivatives

    # stopCB 
    stopCallback(nfmu, c, t_stop)

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
    parameters::Union{Dict{<:Any, <:Any}, Nothing} = nothing,
    setup::Union{Bool, Nothing} = nothing,
    reset::Union{Bool, Nothing} = nothing,
    instantiate::Union{Bool, Nothing} = nothing,
    freeInstance::Union{Bool, Nothing} = nothing,
    terminate::Union{Bool, Nothing} = nothing) where {F, C}

    t_start, t_stop = tspan

    c = (hasCurrentComponent(nfmu.fmu) ? getCurrentComponent(nfmu.fmu) : nothing)
    c, _ = prepareSolveFMU(nfmu.fmu, c, fmi2TypeCoSimulation, instantiate, freeInstance, terminate, reset, setup, parameters, t_start, t_stop, tolerance; cleanup=true)
    
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
                                         parameters::Union{Vector{Union{Dict{<:Any, <:Any}, Nothing}}, Nothing} = nothing,
                                         setup::Union{Bool, Nothing} = nothing,
                                         reset::Union{Bool, Nothing} = nothing,
                                         instantiate::Union{Bool, Nothing} = nothing,
                                         freeInstance::Union{Bool, Nothing} = nothing,
                                         terminate::Union{Bool, Nothing} = nothing) where {F, C}

    t_start, t_stop = tspan
    numFMU = length(nfmu.fmu)

    cs = nothing
    ignore_derivatives() do
        cs = Vector{Union{FMU2Component, Nothing}}(undef, numFMU)
        for i in 1:numFMU
            cs[i] = (hasCurrentComponent(nfmu.fmu[i]) ? getCurrentComponent(nfmu.fmu[i]) : nothing)
        end
    end
    cs, _ = prepareSolveFMU(nfmu.fmu, cs, fmi2TypeCoSimulation, instantiate, freeInstance, terminate, reset, setup, parameters, t_start, t_stop, tolerance; cleanup=true)
    
    solution = FMU2Solution(nothing)

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

    return Flux.params(nfmu.p)
end

function Flux.params(nfmu::CS_NeuralFMU; destructure::Bool=true)
    if destructure 
        nfmu.p, nfmu.re = Flux.destructure(nfmu.model)
        return Flux.params(nfmu.p)
    else
        return Flux.params(nfmu.model)
    end
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

    has_nan = any(collect(any(isnan.(grad)) for grad in grads))
    has_nothing = any(collect(any(isnothing.(grad)) for grad in grads)) || any(isnothing.(grads))

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
                    @info "Grad: Min = $(min(abs.(grad)...))   Max = $(max(abs.(grad)...))"
                    @info "Step: Min = $(min(abs.(step)...))   Max = $(max(abs.(step)...))"
                end
                
            end

        end    

    catch e

        if proceed_on_assert
            @error "Training asserted, but continuing: $e"
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

    train!(loss, params::Union{Flux.Params, Zygote.Params}, data, optim::Flux.Optimise.AbstractOptimiser; gradient::Symbol=:Zygote, cb=nothing, chunk_size::Integer=64, printStep::Bool=false)

A function analogous to Flux.train! but with additional features and explicit parameters (faster).

# Arguments
- `loss` a loss function in the format `loss(p)`
- `params` a object holding the parameters
- `data` the training data (or often an iterator)
- `optim` the optimizer used for training 

# Keywords 
- `gradient` a symbol determining the AD-library for gradient computation, available are `:ForwardDiff`, `:Zygote` and :ReverseDiff (default)
- `cb` a custom callback function that is called after every training step
- `chunk_size` the chunk size for AD using ForwardDiff (ignored for other AD-methods)
- `printStep` a boolean determining wheater the gradient min/max is printed after every step (for gradient debugging)
- `proceed_on_assert` a boolean that determins wheater to throw an ecxeption on error or proceed training and just print the error
- `numThreads` [WIP]: an integer determining how many threads are used for training (how many gradients are generated in parallel)
- `multiObjective`: set this if the loss function returns multiple values (multi objective optimization), currently gradients are fired to the optimizer one after another (default `false`)
"""
function train!(loss, params::Union{Flux.Params, Zygote.Params, AbstractVector{<:AbstractVector{<:Real}}}, data, optim; gradient::Symbol=:ReverseDiff, cb=nothing, chunk_size::Union{Integer, Symbol}=:auto_fmiflux, printStep::Bool=false, proceed_on_assert::Bool=false, multiThreading::Bool=false, multiObjective::Bool=false) # ::Flux.Optimise.AbstractOptimiser

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

"""
    ToDo.
"""
function train!(loss, neuralFMU::Union{ME_NeuralFMU, CS_NeuralFMU}, data, optim::Flux.Optimise.AbstractOptimiser; kwargs...)
    params = Flux.params(neuralFMU)   
    train!(loss, params, data, optim; kwargs...)
end

function train!(loss, params::Union{Flux.Params, Zygote.Params, AbstractVector{<:AbstractVector{<:Real}}}, data, optim::Flux.Optimise.AbstractOptimiser; gradient::Symbol=:ReverseDiff, chunk_size::Union{Integer, Symbol}=:auto_fmiflux, multiObjective::Bool=false, kwargs...) 
    if length(params) <= 0 || length(params[1]) <= 0 
        @warn "train!(...): Empty parameter array, training on an empty parameter array doesn't make sense."
        return 
    end
    
    grad_buffer = nothing

    if multiObjective
        dim = loss(params[1])

        grad_buffer = zeros(Float64, length(params[1]), length(dim))
    else
        grad_buffer = zeros(Float64, length(params[1])) 
    end

    grad_fun! = (G, p) -> computeGradient!(G, loss, p, gradient, chunk_size, multiObjective)
    _optim = FluxOptimiserWrapper(optim, grad_fun!, grad_buffer)
    train!(loss, params, data, _optim; gradient=gradient, chunk_size=chunk_size, multiObjective=multiObjective, kwargs...)
end

function train!(loss, params::Union{Flux.Params, Zygote.Params, AbstractVector{<:AbstractVector{<:Real}}}, data, optim::Optim.AbstractOptimizer; gradient::Symbol=:ReverseDiff, chunk_size::Union{Integer, Symbol}=:auto_fmiflux, multiObjective::Bool=false, kwargs...) 
    if length(params) <= 0 || length(params[1]) <= 0 
        @warn "train!(...): Empty parameter array, training on an empty parameter array doesn't make sense."
        return 
    end
    
    grad_fun! = (G, p) -> computeGradient!(G, loss, p, gradient, chunk_size, multiObjective)
    _optim = OptimOptimiserWrapper(optim, grad_fun!, loss, params[1])
    train!(loss, params, data, _optim; gradient=gradient, chunk_size=chunk_size, multiObjective=multiObjective, kwargs...)
end
