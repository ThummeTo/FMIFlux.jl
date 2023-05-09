#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import ChainRulesCore: ignore_derivatives
import FMIImport: assert_integrator_valid, fd_eltypes, fd_set!, finishSolveFMU,
    handleEvents, isdual, istracked, prepareSolveFMU, rd_set!, undual, unsense, untrack
import Optim
import ProgressMeter
import SciMLSensitivity.SciMLBase: CallbackSet, ContinuousCallback, ODESolution, ReturnCode, RightRootFind,
    VectorContinuousCallback, set_u!, terminate!, u_modified!, build_solution
import SciMLSensitivity.ForwardDiff
import SciMLSensitivity.ReverseDiff
using SciMLSensitivity.ReverseDiff: TrackedArray
import SciMLSensitivity: InterpolatingAdjoint, ReverseDiffVJP
import ThreadPools

using DiffEqCallbacks
using DifferentialEquations: ODEFunction, ODEProblem, solve
using FMIImport: FMU2Component, FMU2Event, FMU2Solution, fmi2ComponentState,
    fmi2ComponentStateContinuousTimeMode, fmi2ComponentStateError,
    fmi2ComponentStateEventMode, fmi2ComponentStateFatal,
    fmi2ComponentStateInitializationMode, fmi2ComponentStateInstantiated,
    fmi2ComponentStateTerminated, fmi2StatusOK, fmi2Type, fmi2TypeCoSimulation,
    fmi2TypeModelExchange, logError, logInfo, logWarn
using Flux
using Flux.Zygote
using SciMLSensitivity:
    ForwardDiffSensitivity, InterpolatingAdjoint, ReverseDiffVJP, ZygoteVJP

zero_tgrad(u,p,t) = zero(u)

"""
The mutable struct representing an abstract (simulation mode unknown) NeuralFMU.
"""
abstract type NeuralFMU end

"""
Structure definition for a NeuralFMU, that runs in mode `Model Exchange` (ME).
"""
mutable struct ME_NeuralFMU{M, P, R} <: NeuralFMU

    model::M
    p::P
    re::R
    kwargs

    fmu::FMU

    tspan
    saveat
    saved_values
    recordValues
    solver

    valueStack

    customCallbacksBefore::Array
    customCallbacksAfter::Array

    x0::Array{Float64}
    firstRun::Bool
    
    tolerance::Union{Real, Nothing}
    parameters::Union{Dict{<:Any, <:Any}, Nothing}
    setup::Union{Bool, Nothing}
    reset::Union{Bool, Nothing}
    instantiate::Union{Bool, Nothing}
    freeInstance::Union{Bool, Nothing}
    terminate::Union{Bool, Nothing}

    modifiedState::Bool

    startState 
    stopState
    startEventInfo
    stopEventInfo
    start_t 
    stop_t

    progressMeter
    execution_start::Real

    function ME_NeuralFMU{M, P, R}(model::M, p::P, re::R) where {M, P, R}
        inst = new()
        inst.model = model 
        inst.p = p 
        inst.re = re 

        inst.progressMeter = nothing
        inst.modifiedState = true

        inst.startState = nothing 
        inst.stopState = nothing

        inst.startEventInfo = nothing 
        inst.stopEventInfo = nothing

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
    saveat

    re # restrucure function

    function CS_NeuralFMU{F, C}() where {F, C}
        inst = new{F, C}()

        inst.re = nothing

        return inst
    end
end

function evaluateModel(nfmu::ME_NeuralFMU, c::FMU2Component, x)
    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    return nfmu.model(x)
end

function evaluateReModel(nfmu::ME_NeuralFMU, c::FMU2Component, x, p)
    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    return nfmu.re(p)(x)
end

##### EVENT HANDLING START

function startCallback(integrator, nfmu::ME_NeuralFMU, c::Union{FMU2Component, Nothing}, t)
    ignore_derivatives() do
        #nfmu.solveCycle += 1
        #@debug "[$(nfmu.solveCycle)][FIRST STEP]"

        nfmu.execution_start = time()

        t = unsense(t)

        @assert t == nfmu.tspan[1] "startCallback(...): Called for non-start-point t=$(t)"
        
        c, x0 = prepareSolveFMU(nfmu.fmu, c, fmi2TypeModelExchange, nfmu.instantiate, nfmu.freeInstance, nfmu.terminate, nfmu.reset, nfmu.setup, nfmu.parameters, nfmu.tspan[1], nfmu.tspan[end], nfmu.tolerance; x0=nfmu.x0, handleEvents=FMIFlux.handleEvents, cleanup=true)
        
        if c.eventInfo.nextEventTime == t && c.eventInfo.nextEventTimeDefined == fmi2True
            @debug "Initial time event detected!"
        else
            @debug "No initial time events ..."
        end

        #@assert fmi2EnterContinuousTimeMode(c) == fmi2StatusOK
    end

    return c
end

function stopCallback(nfmu::ME_NeuralFMU, c::FMU2Component, t)

    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    ignore_derivatives() do
        #@debug "[$(nfmu.solveCycle)][LAST STEP]"

        t = unsense(t)

        @assert t == nfmu.tspan[end] "stopCallback(...): Called for non-start-point t=$(t)"

        #c = finishSolveFMU(nfmu.fmu, c, nfmu.freeInstance, nfmu.terminate)
        
    end

    return c
end

# Read next time event from fmu and provide it to the integrator 
function time_choice(nfmu::ME_NeuralFMU, c::FMU2Component, integrator, tStart, tStop)

    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @debug assert_integrator_valid(integrator)

    # last call may be after simulation end
    if c == nothing
        return nothing
    end

    if !c.fmu.executionConfig.handleTimeEvents
        return nothing
    end

    if c.eventInfo.nextEventTimeDefined == fmi2True
        #@debug "time_choice(...): $(c.eventInfo.nextEventTime) at t=$(ForwardDiff.value(integrator.t))"

        if c.eventInfo.nextEventTime >= tStart && c.eventInfo.nextEventTime <= tStop
            #@assert sizeof(integrator.t) == sizeof(c.eventInfo.nextEventTime) "The NeuralFMU/solver are initialized in $(sizeof(integrator.t))-bit-mode, but FMU events are defined in $(sizeof(c.eventInfo.nextEventTime))-bit. Please define your ANN in $(sizeof(c.eventInfo.nextEventTime))-bit mode."
            @debug "time_choice(...): At $(integrator.t) next time event announced @$(c.eventInfo.nextEventTime)s"
            return c.eventInfo.nextEventTime
        else
            # the time event is outside the simulation range!
            @debug "Next time event @$(c.eventInfo.nextEventTime)s is outside simulation time range ($(tStart), $(tStop)), skipping."
            return nothing 
        end
    else
        #@debug "time_choice(...): nothing at t=$(ForwardDiff.value(integrator.t))"
        return nothing
    end

    
end

# Returns the event indicators for an FMU.
function condition(nfmu::ME_NeuralFMU, c::FMU2Component, out::SubArray{<:ForwardDiff.Dual{T, V, N}, A, B, C, D}, _x, t, integrator) where {T, V, N, A, B, C, D} # Event when event_f(u,t) == 0
    
    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @debug assert_integrator_valid(integrator)

    @assert c.state == fmi2ComponentStateContinuousTimeMode "condition(...): Must be called in mode continuous time."

    # ToDo: set inputs here
    #fmiSetReal(myFMU, InputRef, Value)

    t = undual(t)
    x = undual(_x)

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    #c.t = t # this will auto-set time via fx-call!
    evaluateModel(nfmu, c, x)
    fmi2SetTime(c, t)

    out_tmp = zeros(c.fmu.modelDescription.numberOfEventIndicators)
    fmi2GetEventIndicators!(c, out_tmp)

    fd_set!(out, out_tmp)

    @debug assert_integrator_valid(integrator)

    return nothing
end
function condition(nfmu::ME_NeuralFMU, c::FMU2Component, out::SubArray{<:ReverseDiff.TrackedReal}, _x, t, integrator)
    
    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @debug assert_integrator_valid(integrator)

    @assert c.state == fmi2ComponentStateContinuousTimeMode "condition(...): Must be called in mode continuous time."

    # ToDo: set inputs here
    #fmiSetReal(myFMU, InputRef, Value)

    t = untrack(t)
    x = untrack(_x)

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    #c.t = t # this will auto-set time via fx-call!
    evaluateModel(nfmu, c, x)
    fmi2SetTime(c, t)

    out_tmp = zeros(c.fmu.modelDescription.numberOfEventIndicators)
    fmi2GetEventIndicators!(c, out_tmp)

    rd_set!(out, out_tmp)

    @debug assert_integrator_valid(integrator)

    return nothing
end
function condition(nfmu::ME_NeuralFMU, c::FMU2Component, out, _x, t, integrator) # Event when event_f(u,t) == 0

    @debug assert_integrator_valid(integrator)
    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    @assert c.state == fmi2ComponentStateContinuousTimeMode "condition(...): Must be called in mode continuous time."

    #@debug "State condition..."

    # ToDo: set inputs here
    #fmiSetReal(myFMU, InputRef, Value)

    t = unsense(t)
    x = unsense(_x)

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    #c.t = t # this will auto-set time via fx-call!
    evaluateModel(nfmu, c, x) # evaluate NeuralFMU (set new states)
    fmi2SetTime(c, t)

    fmi2GetEventIndicators!(c, out)

    @debug assert_integrator_valid(integrator)

    return nothing
end

global lastIndicator = nothing
global lastIndicatorX = nothing 
global lastIndicatorT = nothing
function conditionSingle(nfmu::ME_NeuralFMU, c::FMU2Component, index, _x, t, integrator) 

    @assert c.state == fmi2ComponentStateContinuousTimeMode "condition(...): Must be called in mode continuous time."
    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    # ToDo: set inputs here
    #fmiSetReal(myFMU, InputRef, Value)

    if c.fmu.executionConfig.handleEventIndicators != nothing && index ∉ c.fmu.executionConfig.handleEventIndicators
        return 1.0
    end

    t = unsense(t)
    x = unsense(_x)

    global lastIndicator # , lastIndicatorX, lastIndicatorT

    if lastIndicator == nothing || length(lastIndicator) != c.fmu.modelDescription.numberOfEventIndicators
        lastIndicator = zeros(c.fmu.modelDescription.numberOfEventIndicators)
    end

    # ToDo: Input Function
    
    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    #c.t = t # this will auto-set time via fx-call!
    evaluateModel(nfmu, c, x) # evaluate NeuralFMU (set new states)
    fmi2SetTime(c, t)

    fmi2GetEventIndicators!(c, lastIndicator)
    
    return lastIndicator[index]
end

function f_optim(x, nfmu::ME_NeuralFMU, c::FMU2Component, right_x_fmu) # , idx, direction::Real)
    # propagete the new state-guess `x` through the NeuralFMU
    evaluateModel(nfmu, c, x)
    #indicators = fmi2GetEventIndicators(c)
    return Flux.Losses.mse(right_x_fmu, fmi2GetContinuousStates(c)) # - min(-direction*indicators[idx], 0.0)
end

# Handles the upcoming events.
function affectFMU!(nfmu::ME_NeuralFMU, c::FMU2Component, integrator, idx)

    @assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @debug assert_integrator_valid(integrator)

    @assert c.state == fmi2ComponentStateContinuousTimeMode "affectFMU!(...): Must be in continuous time mode!"

    t = unsense(integrator.t)
    x = unsense(integrator.u)

    # there are fx-evaluations before the event is handled, reset the FMU state to the current integrator step
    mode = c.force
    c.force = true

    evaluateModel(nfmu, c, x) # evaluate NeuralFMU (set new states)
    fmi2SetTime(c, t)

    c.force = mode

    # if inputFunction !== nothing
    #     fmi2SetReal(c, inputValues, inputFunction(integrator.t))
    # end

    fmi2EnterEventMode(c)

    #############

    #left_x_fmu = fmi2GetContinuousStates(c)
    #fmi2SetTime(c, t)
    # Todo set inputs

    # Event found - handle it
    #@assert fmi2EnterEventMode(c) == fmi2StatusOK
    handleEvents(c)

    ignore_derivatives() do
        if idx == 0
            #@debug "affectFMU!(...): Handle time event at t=$t"
        end

        if idx > 0
            #@debug "affectFMU!(...): Handle state event at t=$t"
        end
    end

    left_x = nothing
    right_x = nothing

    if c.eventInfo.valuesOfContinuousStatesChanged == fmi2True

        left_x = x

        right_x_fmu = fmi2GetContinuousStates(c) # the new FMU state after handled events

        ignore_derivatives() do 
            #@debug "affectFMU!(_, _, $idx): NeuralFMU state event from $(left_x) (fmu: $(left_x_fmu)). Indicator [$idx]: $(indicators[idx]). Optimizing new state ..."
        end

        # ToDo: Problem-related parameterization of optimize-call
        #result = optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu), left_x, LBFGS(); autodiff = :forward)
        #result = Optim.optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu, idx, sign(indicators[idx])), left_x, NelderMead())
        
        # if there is an ANN above the FMU, propaget FMU state through top ANN:
        if nfmu.modifiedState == true
            result = Optim.optimize(x_seek -> f_optim(x_seek, nfmu, c, right_x_fmu), left_x, NelderMead())
            right_x = Optim.minimizer(result)
        else # if there is no ANN above, then:
            right_x = right_x_fmu
        end

        if isdual(integrator.u)
            T, V, N = fd_eltypes(integrator.u)

            new_x = collect(ForwardDiff.Dual{T, V, N}(V(right_x[i]), ForwardDiff.partials(integrator.u[i]))   for i in 1:length(integrator.u))
            set_u!(integrator, new_x)
            
            @debug "affectFMU!(_, _, $idx): NeuralFMU event with state change at $t. Indicator [$idx]. (ForwardDiff) "
        else
            integrator.u = right_x
            @debug "affectFMU!(_, _, $idx): NeuralFMU event with state change at $t. Indicator [$idx]."
        end
       
        u_modified!(integrator, true)
    else

        ignore_derivatives() do 
            @debug "affectFMU!(_, _, $idx): NeuralFMU event without state change at $t. Indicator [$idx]."
        end

        u_modified!(integrator, false)
    end

    if c.eventInfo.nominalsOfContinuousStatesChanged == fmi2True
        x_nom = fmi2GetNominalsOfContinuousStates(c)
    end

    ignore_derivatives() do
        if idx != -1
            e = FMU2Event(t, UInt64(idx), left_x, right_x)
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
            logError(c.fmu, "Event jittering detected $(round(Integer, ratio)) events/s, aborting at t=$(t) (rel. t=$(pt)) at event $(ne):")
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

    @debug assert_integrator_valid(integrator)
end

# Does one step in the simulation.
function stepCompleted(nfmu::ME_NeuralFMU, c::FMU2Component, x, t, integrator, tStart, tStop)

    #@assert getCurrentComponent(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @debug assert_integrator_valid(integrator)

    #@debug "Step"
    # there might be no component (in Zygote)!
    # @assert c.state == fmi2ComponentStateContinuousTimeMode "stepCompleted(...): Must be in continuous time mode."

    if nfmu.progressMeter !== nothing
        ProgressMeter.update!(nfmu.progressMeter, floor(Integer, 1000.0*(t-tStart)/(tStop-tStart)) )
    end

    if c != nothing
        (status, enterEventMode, terminateSimulation) = fmi2CompletedIntegratorStep(c, fmi2True)

        if terminateSimulation == fmi2True
            logError(c.fmu, "stepCompleted(...): FMU requested termination!")
        end

        if enterEventMode == fmi2True
            affectFMU!(nfmu, c, integrator, -1)
        else
            # ToDo: set inputs here
            #fmiSetReal(myFMU, InputRef, Value)
        end

        #@debug "Step completed at $(ForwardDiff.value(t)) with $(collect(ForwardDiff.value(xs) for xs in x))"
    end

    @debug assert_integrator_valid(integrator)
end

# save FMU values 
function saveValues(nfmu::ME_NeuralFMU, c::FMU2Component, recordValues, _x, t, integrator)

    t = unsense(t) 
    x = unsense(_x)

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    #c.t = t # this will auto-set time via fx-call!
    evaluateModel(nfmu, c, x) # evaluate NeuralFMU (set new states)
    fmi2SetTime(c, t)
    
    # Todo set inputs
    
    return (fmi2GetReal(c, recordValues)...,)
end

function fx(nfmu::ME_NeuralFMU,
    c::FMU2Component,
    dx,#::Array{<:Real},
    x,#::Array{<:Real},
    p,#::Array,
    t)#::Real) 

    #nanx = !any(isnan.(collect(any(isnan.(ForwardDiff.partials.(x[i]).values)) for i in 1:length(x))))
    #nanu = !any(isnan(ForwardDiff.partials(t)))

    #@assert nanx && nanu "NaN in start fx nanx = $nanx   nanu = $nanu @ $(t)."
    
    dx_tmp = fx(nfmu,c,x,p,t)

    if isdual(dx)
        fd_set!(dx, dx_tmp)
    elseif istracked(dx)
        rd_set!(dx, dx_tmp)
    else
        #@info "dx: $(dx)"
        #@info "dx_tmp: $(dx_tmp)"
        dx[:] = dx_tmp[:]
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

    #c.t = t 
    dx = evaluateReModel(nfmu, c, x, p)

    ignore_derivatives() do

        t = unsense(t)
        fmi2SetTime(c, t)

        #@debug "fx($t, $(collect(ForwardDiff.value(xs) for xs in x))) = $(collect(ForwardDiff.value(xs) for xs in dx))"

        #c.solution.evals_fx += 1
    end 

    return dx
end

##### EVENT HANDLING END

"""
Constructs a ME-NeuralFMU where the FMU is at an arbitrary location inside of the NN.

# Arguents
    - `fmu` the considered FMU inside the NN 
    - `model` the NN topology (e.g. Flux.chain)
    - `tspan` simulation time span
    - `alg` a numerical ODE solver
    - `convertParams` automatically convert ANN parameters to Float64 if not already

# Keyword arguments
    - `saveat` time points to save the NeuralFMU output, if empty, solver step size is used (may be non-equidistant)
    - `fixstep` forces fixed step integration
    - `recordFMUValues` additionally records internal FMU variables (currently not supported because of open issues)
"""
function ME_NeuralFMU(fmu::FMU2, 
                      model, 
                      tspan, 
                      solver=nothing; 
                      saveat=[], 
                      recordValues = nothing, 
                      kwargs...)

    if !is64(model)
        model = convert64(model)
    end
    
    p, re = Flux.destructure(model)
    nfmu = ME_NeuralFMU{typeof(model), typeof(p), typeof(re)}(model, p, re)

    ######

    nfmu.fmu = fmu
    
    nfmu.saved_values = nothing

    nfmu.recordValues = prepareValueReference(fmu, recordValues)

    # abstol=abstol, reltol=reltol, dtmin=dtmin, force_dtmin=force_dtmin, 
    
    nfmu.tspan = tspan
    nfmu.saveat = saveat
    nfmu.solver = solver
    nfmu.kwargs = kwargs
    nfmu.parameters = nothing

    ######
    
    nfmu
end

"""
Constructs a CS-NeuralFMU where the FMU is at an arbitrary location inside of the NN.

# Arguents
    - `fmu` the considered FMU inside the NN 
    - `model` the NN topology (e.g. Flux.chain)
    - `tspan` simulation time span

# Keyword arguments
    - `saveat` time points to save the NeuralFMU output, if empty, solver step size is used (may be non-equidistant)
"""
function CS_NeuralFMU(fmu::Union{FMU2, Vector{<:FMU2}},
                      model, 
                      tspan; 
                      saveat=[], 
                      recordValues = [])

    if !is64(model)
        model = convert64(model)
    end

    nfmu = nothing
    if typeof(fmu) == FMU2
        nfmu = CS_NeuralFMU{FMU2, FMU2Component}()
    else
        nfmu = CS_NeuralFMU{Vector{FMU2}, Vector{FMU2Component} }()
    end

    nfmu.fmu = fmu

    nfmu.model = model # Chain(model.layers...)

    nfmu.tspan = tspan
    nfmu.saveat = saveat

    nfmu
end

function checkExecTime(integrator, nfmu::ME_NeuralFMU, c, max_execution_duration::Real)
    dist = max(nfmu.execution_start + max_execution_duration - time(), 0.0)
    
    if dist <= 0.0
        logWarn(nfmu.fmu, "Reached max execution duration ($(max_execution_duration)), exiting...")
        terminate!(integrator)
    end

    return dist
end

"""
Evaluates the ME_NeuralFMU in the timespan given during construction or in a custom timespan from `t_start` to `t_stop` for a given start state `x_start`.

# Keyword arguments
    - `reset`, the FMU is reset every time evaluation is started (default=`true`).
    - `setup`, the FMU is set up every time evaluation is started (default=`true`).
"""
function (nfmu::ME_NeuralFMU)(x_start::Union{Array{<:Real}, Nothing} = nfmu.x0,
    tspan::Tuple{Float64, Float64} = nfmu.tspan;
    showProgress::Bool = false,
    progressDescr::String="Simulating ME-NeuralFMU ...",
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
    saveat=nfmu.saveat, # ToDo: Data type 
    kwargs...)

    @assert saveat[1] == tspan[1] "NeuralFMU changed time interval, start time is $(tspan[1]), but saveat from constructor gives $(saveat[1]). Please provide correct `saveat` via keyword with matching start/stop time."
    @assert saveat[end] == tspan[end] "NeuralFMU changed time interval, stop time is $(tspan[end]), but saveat from constructor gives $(saveat[end]). Please provide correct `saveat` via keyword with matching start/stop time."

    recordValues = prepareValueReference(nfmu.fmu, recordValues)

    saving = (length(recordValues) > 0)
    sense = nfmu.fmu.executionConfig.sensealg
    inPlace = nfmu.fmu.executionConfig.inPlace

    t_start = tspan[1]
    t_stop = tspan[end]

    nfmu.tspan = tspan
    nfmu.x0 = x_start

    ignore_derivatives() do
        @debug "ME_NeuralFMU..."

        nfmu.firstRun = true

        nfmu.tolerance = tolerance

        @info "$(parameters)"
        if isnothing(parameters)
            if !isnothing(nfmu.fmu.optim_p_refs)
                nfmu.parameters = Dict(nfmu.fmu.optim_p_refs .=> unsense(nfmu.fmu.optim_p))
            end
        else
            nfmu.parameters = parameters
        end
        nfmu.setup = setup
        nfmu.reset = reset
        nfmu.instantiate = instantiate
        nfmu.freeInstance = freeInstance
        nfmu.terminate = terminate
       
        nfmu.progressMeter = nothing
    end

    callbacks = []

    c = (hasCurrentComponent(nfmu.fmu) ? getCurrentComponent(nfmu.fmu) : nothing)
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
            if c.fmu.executionConfig.useVectorCallbacks

                eventCb = VectorContinuousCallback((out, x, t, integrator) -> condition(nfmu, c, out, x, t, integrator),
                                                (integrator, idx) -> affectFMU!(nfmu, c, integrator, idx),
                                                Int64(c.fmu.modelDescription.numberOfEventIndicators);
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
            #@info "Setting max execeution time to $(max_execution_duration)"
        end

        # custom callbacks
        for cb in nfmu.customCallbacksAfter
            push!(callbacks, cb)
        end

        if showProgress
            nfmu.progressMeter = ProgressMeter.Progress(1000; desc=progressDescr, color=:blue, dt=1.0) #, barglyphs=ProgressMeter.BarGlyphs("[=> ]"))
            ProgressMeter.update!(nfmu.progressMeter, 0) # show it!
        end

        # integrator step callback
        stepCb = FunctionCallingCallback((x, t, integrator) -> stepCompleted(nfmu, c, x, t, integrator, t_start, t_stop);
                                            func_everystep=true,
                                            func_start=true)
        push!(callbacks, stepCb)

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

    end # ignore_derivatives

    prob = nothing

    if inPlace
        ff = ODEFunction{true}((dx, x, p, t) -> fx(nfmu, c, dx, x, p, t), 
                               tgrad=nothing)
        prob = ODEProblem{true}(ff, nfmu.x0, nfmu.tspan, p)
    else 
        ff = ODEFunction{false}((x, p, t) -> fx(nfmu, c, x, p, t), 
                                tgrad=nothing) # zero_tgrad)
        prob = ODEProblem{false}(ff, nfmu.x0, nfmu.tspan, p)
    end

    if isnothing(sense)
        sense = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))
    end

    # if (length(callbacks) == 2) # only start and stop callback, so the system is pure continuous
    #     startCallback(nfmu, nfmu.tspan[1])
    #     c.solution.states = solve(prob, nfmu.args...; sensealg=sense, saveat=nfmu.saveat, nfmu.kwargs...)
    #     stopCallback(nfmu, nfmu.tspan[end])
    # else
    #c.solution.states = solve(prob, nfmu.args...; sensealg=sense, saveat=nfmu.saveat, callback = CallbackSet(callbacks...), nfmu.kwargs...)

    kwargs = Dict{Symbol, Any}(kwargs...)

    if !isnothing(saveat)
        kwargs[:saveat] = saveat
    end
        
    c.solution.states = solve(prob, nfmu.solver; sensealg=sense, callback=CallbackSet(callbacks...), nfmu.kwargs..., kwargs...) 

    #end

    ignore_derivatives() do
        # ReverseDiff returns an array instead of an ODESolution, this needs to be corrected
        if isa(c.solution.states, TrackedArray)
           
            t = collect(saveat)
            u = c.solution.states
            c.solution.success = (size(u)[2] == length(t)) 

            if c.solution.success
                c.solution.states = build_solution(prob, nfmu.solver, t, collect(u[:,i] for i in 1:size(u)[2]))
            end
        else
            c.solution.success = (c.solution.states.retcode == ReturnCode.Success)
        end
        
    end # ignore_derivatives

    # stopCB (Opt B)
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
Evaluates the CS_NeuralFMU in the timespan given during construction or in a custum timespan from `t_start` to `t_stop` with a given time step size `t_step`.

Via optional argument `reset`, the FMU is reset every time evaluation is started (default=`true`).
"""
function (nfmu::CS_NeuralFMU{F, C})(inputFct,
    t_step::Real, 
    tspan::Tuple{Float64, Float64} = nfmu.tspan; 
    p=nothing,
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
    #c.skipNextDoStep = true # skip first fim2DoStep-call
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
    # for c in cs
    #     c.skipNextDoStep = true
    # end
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
            #_p = collect(ForwardDiff.value(r) for r in p[1])
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
        p, nfmu.re = Flux.destructure(nfmu.model)
        return Flux.params(p)
    else
        return Flux.params(nfmu.model)
    end
end

# FMI version independent dosteps

# function ChainRulesCore.rrule(f::Union{typeof(fmi2SetupExperiment), 
#                                        typeof(fmi2EnterInitializationMode), 
#                                        typeof(fmi2ExitInitializationMode),
#                                        typeof(fmi2Reset),
#                                        typeof(fmi2Terminate)}, args...)

#     y = f(args...)

#     function pullback(ȳ)
#         return collect(ZeroTangent() for arg in args)
#     end

#     return y, fmi2EvaluateME_pullback
# end

function computeGradient(loss, params, gradient, chunk_size)

    if gradient == :ForwardDiff

        if chunk_size == :auto_forwarddiff
            
            grad_conf = ForwardDiff.GradientConfig(loss, params);
            return ForwardDiff.gradient(loss, params, grad_conf);

        elseif chunk_size == :auto_fmiflux

            chunk_size = 32
            grad_conf = ForwardDiff.GradientConfig(loss, params, ForwardDiff.Chunk{min(chunk_size, length(params))}());
            return ForwardDiff.gradient(loss, params, grad_conf);
        else

            grad_conf = ForwardDiff.GradientConfig(loss, params, ForwardDiff.Chunk{min(chunk_size, length(params))}());
            return ForwardDiff.gradient(loss, params, grad_conf);
        end

    elseif gradient == :Zygote 

        return Zygote.gradient(
            loss,
            params)[1]
    elseif gradient == :ReverseDiff 

        return ReverseDiff.gradient(
                loss,
                params)
    else
        @assert false "Unknown `gradient=$(gradient)`, supported are `:ForwardDiff`, `:Zygote` and `:ReverseDiff`."
    end

end

function trainStep(loss, params, gradient, chunk_size, optim, printStep, proceed_on_assert, cb)
    try
                
        for j in 1:length(params)

            grad = computeGradient(loss, params[j], gradient, chunk_size)
            
            @assert !isnothing(grad) "Gradient nothing!"

            step = Flux.Optimise.apply!(optim, params[j], grad)
            params[j] .-= step

            if printStep
                @info "Grad: Min = $(min(abs.(grad)...))   Max = $(max(abs.(grad)...))"
                @info "Step: Min = $(min(abs.(step)...))   Max = $(max(abs.(step)...))"
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
"""
function train!(loss, params::Union{Flux.Params, Zygote.Params}, data, optim::Flux.Optimise.AbstractOptimiser; gradient::Symbol=:ReverseDiff, cb=nothing, chunk_size::Union{Integer, Symbol}=:auto_forwarddiff, printStep::Bool=false, proceed_on_assert::Bool=false, multiThreading::Bool=false)

    if multiThreading && Threads.nthreads() == 1 
        @warn "train!(...): Multi-threading is set via flag `multiThreading=true`, but this Julia process does not have multiple threads. This will not result in a speed-up. Please spawn Julia in multi-thread mode to speed-up training."
    end

    if length(params) <= 0 || length(params[1]) <= 0 
        @warn "train!(...): Empty parameter array, training on an empty parameter array doesn't make sense."
        return 
    end

    _trainStep = (i,) -> trainStep(loss, params, gradient, chunk_size, optim, printStep, proceed_on_assert, cb)

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