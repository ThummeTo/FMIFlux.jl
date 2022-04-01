#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Zygote
using Zygote: @adjoint

using Flux, DiffEqFlux
using OrdinaryDiffEq
using DiffEqCallbacks
using Interpolations: interpolate, LinearInterpolation

using DiffEqFlux: ODEFunction, basic_tgrad, ODEProblem, ZygoteVJP, InterpolatingAdjoint, solve

import ForwardDiff
import Optim
import ProgressMeter

import SciMLBase: RightRootFind

using FMIImport: fmi2ComponentState, fmi2ComponentStateInstantiated, fmi2ComponentStateInitializationMode, fmi2ComponentStateEventMode, fmi2ComponentStateContinuousTimeMode, fmi2ComponentStateTerminated, fmi2ComponentStateError, fmi2ComponentStateFatal
import ChainRulesCore: ignore_derivatives

using FMIImport: fmi2StatusOK

"""
The mutable struct representing an abstract (simulation mode unknown) NeuralFMU.
"""
abstract type NeuralFMU end

"""
Structure definition for a NeuralFMU, that runs in mode `Model Exchange` (ME).
"""
mutable struct ME_NeuralFMU <: NeuralFMU
    neuralODE::NeuralODE
    solution::ODESolution
    fmu::FMU

    currentComponent

    tspan
    saveat
    saved_values
    recordValues

    valueStack

    customCallbacks::Array

    x0::Array{Float64}
    firstRun::Bool
   
    function ME_NeuralFMU()
        inst = new()
        inst.currentComponent = nothing

        return inst 
    end
end

"""
Structure definition for a NeuralFMU, that runs in mode `Co-Simulation` (CS).
"""
mutable struct CS_NeuralFMU{F, C} <: NeuralFMU
    model
    fmu::F
    currentComponent::C

    tspan
    saveat
    valueStack

    function CS_NeuralFMU{F, C}() where {F, C} 
        inst = new{F, C}()

        return inst
    end
end

##### EVENT HANDLING START

# Read next time event from fmu and provide it to the integrator 
function time_choice(c::FMU2Component, integrator)
   
    #@debug "Time condition..."

    if c.eventInfo.nextEventTimeDefined == fmi2True
        ignore_derivatives() do 
            #@debug "time_choice(_, _): Next event defined at $(eventInfo.nextEventTime)s"
        end 
        return c.eventInfo.nextEventTime
    else
        return nothing
    end
end

# Handles events and returns the values and nominals of the changed continuous states.
function handleEvents(c::FMU2Component)

    @assert c.state == fmi2ComponentStateEventMode "handleEvents(...): Must be in event mode!"

    #@debug "Handle Events..."

    # trigger the loop
    c.eventInfo.newDiscreteStatesNeeded = fmi2True

    valuesOfContinuousStatesChanged = fmi2False
    nominalsOfContinuousStatesChanged = fmi2False
    nextEventTimeDefined = fmi2False
    nextEventTime = 0.0

    while c.eventInfo.newDiscreteStatesNeeded == fmi2True
        fmi2NewDiscreteStates!(c, c.eventInfo)

        if c.eventInfo.valuesOfContinuousStatesChanged == fmi2True
            valuesOfContinuousStatesChanged = fmi2True 
        end 

        if c.eventInfo.nominalsOfContinuousStatesChanged == fmi2True
            nominalsOfContinuousStatesChanged = fmi2True
        end

        if c.eventInfo.nextEventTimeDefined == fmi2True
            nextEventTimeDefined = fmi2True
            nextEventTime = c.eventInfo.nextEventTime
        end

        if c.eventInfo.terminateSimulation == fmi2True
            @error "handleEvents(...): FMU throws `terminateSimulation`!"
        end
    end

    c.eventInfo.valuesOfContinuousStatesChanged = valuesOfContinuousStatesChanged
    c.eventInfo.nominalsOfContinuousStatesChanged = nominalsOfContinuousStatesChanged
    c.eventInfo.nextEventTimeDefined = nextEventTimeDefined
    c.eventInfo.nextEventTime = nextEventTime

    return nothing
end

# Returns the event indicators for an FMU.
function condition(nfmu::ME_NeuralFMU, out, x, t, integrator) # Event when event_f(u,t) == 0

    @assert nfmu.fmu.components[end].state == fmi2ComponentStateContinuousTimeMode "condition(...): Must be called in mode continuous time."

    #@debug "State condition..."

    # ToDo: set inputs here
    #fmiSetReal(myFMU, InputRef, Value)

    if isa(t, ForwardDiff.Dual) 
        t = ForwardDiff.value(t)
    end 

    if all(isa.(x, ForwardDiff.Dual))
        x = collect(ForwardDiff.value(e) for e in x)
    end

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    nfmu.neuralODE.model(x) # evaluate NeuralFMU (set new states)
    fmi2SetTime(nfmu.fmu, t)

    # ToDo: Input Function

    if all(isa.(out, ForwardDiff.Dual))
        buf = fmi2GetEventIndicators(nfmu.fmu.components[end])
        T, V, N = fd_eltypes(out)
        out[:] = collect(ForwardDiff.Dual{T, V, N}(V(buf[i]), ForwardDiff.partials(out[i])    ) for i in 1:length(out))
    else 
        fmi2GetEventIndicators!(nfmu.fmu.components[end], out)
    end
    
    return nothing
end
function conditionSingle(nfmu::ME_NeuralFMU, index, x, t, integrator) 

    @assert nfmu.fmu.components[end].state == fmi2ComponentStateContinuousTimeMode "condition(...): Must be called in mode continuous time."

    # ToDo: set inputs here
    #fmiSetReal(myFMU, InputRef, Value)

    if isa(t, ForwardDiff.Dual) 
        t = ForwardDiff.value(t)
    end 

    if all(isa.(x, ForwardDiff.Dual))
        x = collect(ForwardDiff.value(e) for e in x)
    end

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    nfmu.neuralODE.model(x) # evaluate NeuralFMU (set new states)
    fmi2SetTime(nfmu.fmu, t)

    # ToDo: Input Function

    indicators = fmi2GetEventIndicators(nfmu.fmu)

    return indicators[index]
end

function f_optim(x, nfmu, right_x_fmu) # , idx, direction::Real)
    # propagete the new state-guess `x` through the NeuralFMU
    nfmu.neuralODE.model(x)
    #indicators = fmi2GetEventIndicators(nfmu.fmu)
    return Flux.Losses.mse(right_x_fmu, fmi2GetContinuousStates(nfmu.fmu)) # - min(-direction*indicators[idx], 0.0)
end

# Handles the upcoming events.
function affectFMU!(nfmu::ME_NeuralFMU, integrator, idx)

    #@debug "Affect FMU..."

    c = nfmu.fmu.components[end]

    @assert c.state == fmi2ComponentStateContinuousTimeMode "affectFMU!(...): Must be in continuous time mode!"

    t = integrator.t
    if isa(t, ForwardDiff.Dual) 
        t = ForwardDiff.value(t)
    end 

    x = integrator.u
    if all(isa.(x, ForwardDiff.Dual))
        x = collect(ForwardDiff.value(e) for e in x)
    end

    left_x_fmu = fmi2GetContinuousStates(c)
    fmi2SetTime(c, t)
    # Todo set inputs 

    fmi2EnterEventMode(c)

    # Event found - handle it
    handleEvents(c)

    if c.eventInfo.valuesOfContinuousStatesChanged == fmi2True

        left_x = x

        right_x_fmu = fmi2GetContinuousStates(c) # the new FMU state after handled events

        ignore_derivatives() do 
            #@debug "affectFMU!(_, _, $idx): NeuralFMU state event from $(left_x) (fmu: $(left_x_fmu)). Indicator [$idx]: $(indicators[idx]). Optimizing new state ..."
        end

        # ToDo: Problem-related parameterization of optimize-call
        #result = optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu), left_x, LBFGS(); autodiff = :forward)
        #result = Optim.optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu, idx, sign(indicators[idx])), left_x, NelderMead())
        result = Optim.optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu), left_x, NelderMead())

        #display(result)

        right_x = Optim.minimizer(result)
        integrator.u = right_x

        ignore_derivatives() do 
            #@debug "affectFMU!(_, _, $idx): NeuralFMU state event to   $(right_x) (fmu: $(right_x_fmu)). Indicator [$idx]: $(fmi2GetEventIndicators(nfmu.fmu)[idx]). Minimum: $(Optim.minimum(result))."
        end
    end

    if c.eventInfo.nominalsOfContinuousStatesChanged == fmi2True
        x_nom = fmi2GetNominalsOfContinuousStates(nfmu.fmu)
    end

    fmi2EnterContinuousTimeMode(c)
end

# Does one step in the simulation.
function stepCompleted(nfmu::ME_NeuralFMU, x, t, integrator, progressMeter, tStart, tStop)

    # there might be no component!
    # @assert nfmu.currentComponent.state == fmi2ComponentStateContinuousTimeMode "stepCompleted(...): Must be in continuous time mode."

    if progressMeter !== nothing 
        ProgressMeter.update!(progressMeter, floor(Integer, 1000.0*(t-tStart)/(tStop-tStart)) )
    end

    if nfmu.currentComponent != nothing
        (status, enterEventMode, terminateSimulation) = fmi2CompletedIntegratorStep(nfmu.currentComponent, fmi2True)

        if enterEventMode == fmi2True
            affectFMU!(nfmu, integrator, 0)
        else
            # ToDo: set inputs here
            #fmiSetReal(myFMU, InputRef, Value)
        end
    end

    ignore_derivatives() do
        
        if isa(t, ForwardDiff.Dual) 
            t = ForwardDiff.value(t)
        end 

        if t == nfmu.tspan[1] # FIRST STEP 
            @debug "[FIRST STEP]"

            # for the first run, there is already a well-setup instance
            if !nfmu.firstRun

                if nfmu.fmu.executionConfig.instantiate

                    # remove old one if we missed it (callback)
                    if nfmu.currentComponent != nothing
                        if nfmu.fmu.executionConfig.freeInstance
                            fmi2FreeInstance!(nfmu.currentComponent)
                        end
                        nfmu.currentComponent = nothing
                    end

                    nfmu.currentComponent = fmi2Instantiate!(nfmu.fmu)
                    @debug "[NEW INST]"
                else
                    nfmu.currentComponent = nfmu.fmu.components[end]
                end

                # soft terminate (if necessary)
                if nfmu.fmu.executionConfig.terminate
                    fmi2Terminate(nfmu.currentComponent; soft=true)
                end

                # soft reset (if necessary)
                if nfmu.fmu.executionConfig.reset
                    fmi2Reset(nfmu.currentComponent; soft=true)
                end
                
                fmi2SetupExperiment(nfmu.currentComponent, nfmu.tspan[1], nfmu.tspan[end])
                fmi2EnterInitializationMode(nfmu.currentComponent)
                fmi2SetContinuousStates(nfmu.currentComponent, nfmu.x0)
                fmi2ExitInitializationMode(nfmu.currentComponent)
                
                handleEvents(nfmu.currentComponent) 
                fmi2EnterContinuousTimeMode(nfmu.currentComponent)
            end
        end
        
        if t == nfmu.tspan[end]
            @debug "[LAST STEP]"
            nfmu.firstRun = false

            if nfmu.fmu.ẋ_interp === nothing

                if nfmu.fmu.executionConfig.useCachedDersSense
                    nfmu.fmu.ẋ_interp = LinearInterpolation(nfmu.fmu.t_cache, nfmu.fmu.ẋ_cache)
                else
                    nfmu.fmu.ẋ_interp = t -> integrator.sol(t, Val{1}; continuity=:right)
                end
                @debug "Interp. Polynominal ready..."
            end

            if nfmu.currentComponent != nothing
                if nfmu.fmu.executionConfig.freeInstance
                    fmi2FreeInstance!(nfmu.currentComponent)
                end

                nfmu.currentComponent = nothing
            end

        end
    end # ignore_derivatives 
end

# save FMU values 
function saveValues(nfmu, c::FMU2Component, recordValues, x, t, integrator)

    if isa(t, ForwardDiff.Dual) 
        t = ForwardDiff.value(t)
    end 
    
    if all(isa.(x, ForwardDiff.Dual))
        x = collect(ForwardDiff.value(e) for e in x)
    end

    fmi2SetTime(c, t)
    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    nfmu.neuralODE.model(x) # evaluate NeuralFMU (set new states)

    (fmi2GetReal(c, recordValues)...,)
end

function fd_eltypes(e::SubArray{<:ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    return (T, V, N)
end

function fx(nfmu,
    dx,#::Array{<:Real},
    x,#::Array{<:Real},
    p::Array,
    t::Real) 
    
    dx_tmp = fx(nfmu,x,p,t)

    if all(isa.(dx_tmp, ForwardDiff.Dual))
        if all(isa.(dx, ForwardDiff.Dual))
            dx[:] = dx_tmp
        else 
            dx[:] = collect(ForwardDiff.value(e) for e in dx_tmp)
        end
        #dx[:] = collect(ForwardDiff.value(e) for e in dx_tmp)
    else 
        if all(isa.(dx, ForwardDiff.Dual))
            dx_tmp = collect(ForwardDiff.value(e) for e in dx)
            fmi2GetDerivatives!(c, dx_tmp)
            T, V, N = fd_eltypes(dx)
            dx[:] = collect(ForwardDiff.Dual{T, V, N}(dx_tmp[i], ForwardDiff.partials(dx[i])    ) for i in 1:length(dx))
        else
            dx[:] = dx_tmp
        end
    end

    #return dx_tmp
    return dx
end

function fx(nfmu,
    x,#::Array{<:Real},
    p::Array,
    t::Real) 
    
    # if all(isa.(x, ForwardDiff.Dual))
    #     x = collect(ForwardDiff.value(e) for e in x)
    # end

    ignore_derivatives() do

        c = nfmu.fmu.components[end]
        
        if isa(t, ForwardDiff.Dual) 
            t = ForwardDiff.value(t)
        end 

        fmi2SetTime(c, t)

    end 

    dx = nfmu.neuralODE.re(p)(x)

    # build up ẋ interpolation polynominal
    ignore_derivatives() do
        if nfmu.fmu.executionConfig.useCachedDersSense
            if nfmu.fmu.ẋ_interp == nothing
                if t <= nfmu.tspan[end]
                    if length(nfmu.fmu.t_cache) == 0 || nfmu.fmu.t_cache[end] < t
                        push!(nfmu.fmu.ẋ_cache, collect(ForwardDiff.value(e) for e in dx) )
                        push!(nfmu.fmu.t_cache, t)
                    end
                end

                if t >= nfmu.tspan[end] # endpoint
                    
                end
            end
        end
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

# Keyword arguments
    - `saveat` time points to save the NeuralFMU output, if empty, solver step size is used (may be non-equidistant)
    - `fixstep` forces fixed step integration
    - `recordFMUValues` additionally records internal FMU variables (currently not supported because of open issues)
"""
function ME_NeuralFMU(fmu::FMU2, 
                      model, 
                      tspan, 
                      alg=nothing; 
                      saveat=[], 
                      recordFMUValues = nothing, 
                      callbacks = [],
                      kwargs...)

    nfmu = ME_NeuralFMU()
    nfmu.fmu = fmu
    nfmu.customCallbacks = callbacks

    nfmu.saved_values = nothing

    nfmu.recordValues = prepareValueReference(fmu, recordFMUValues)

    nfmu.neuralODE = NeuralODE(model, tspan, alg; saveat=saveat, kwargs...)

    nfmu.tspan = tspan
    nfmu.saveat = saveat

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

"""
Evaluates the ME_NeuralFMU in the timespan given during construction or in a custom timespan from `t_start` to `t_stop` for a given start state `x_start`.

# Keyword arguments
    - `reset`, the FMU is reset every time evaluation is started (default=`true`).
    - `setup`, the FMU is set up every time evaluation is started (default=`true`).
"""
function (nfmu::ME_NeuralFMU)(x_start::Union{Array{<:Real}, Nothing} = nothing, 
                              t_start::Union{Real, Nothing} = nothing, 
                              t_stop::Union{Real, Nothing} = nothing;
                              showProgress::Bool = false, 
                              kwargs...)
    ignore_derivatives() do
        @debug "ME_NeuralFMU..."
    end

    saving = (length(nfmu.recordValues) > 0)
    
    nfmu.fmu.t_cache = [] 
    nfmu.fmu.ẋ_cache = []
    nfmu.fmu.ẋ_interp = nothing
    nfmu.firstRun = true

    ########

    callbacks = []
    sense = nfmu.fmu.executionConfig.sensealg
    inPlace = nfmu.fmu.executionConfig.inPlace

    tspan = getfield(nfmu.neuralODE,:tspan)
    if t_start === nothing
        t_start = tspan[1]
    end
    if t_stop === nothing
        t_stop = tspan[end]
    end
    nfmu.tspan = (t_start, t_stop)

    #################

    progressMeter = nothing

    ignore_derivatives() do

        if nfmu.fmu.executionConfig.instantiate

            nfmu.currentComponent = fmi2Instantiate!(nfmu.fmu)
            @debug "[NEW INST]"
        else
            @assert length(nfmu.fmu.components) > 0 "ME_NeuralFMU(...): No components allocated, auto allocating disabled."
            nfmu.currentComponent = nfmu.fmu.components[end]
        end

        # soft terminate (if necessary)
        if nfmu.fmu.executionConfig.terminate
            fmi2Terminate(nfmu.currentComponent; soft=true)
        end

        # soft reset (if necessary)
        if nfmu.fmu.executionConfig.reset
            fmi2Reset(nfmu.currentComponent; soft=true)
        end
        
        # setup (hard) start
        if nfmu.fmu.executionConfig.setup
            fmi2SetupExperiment(nfmu.currentComponent, nfmu.tspan[1], nfmu.tspan[end])
            fmi2EnterInitializationMode(nfmu.currentComponent)
        end 

        # from here on, we are in initialization mode, if `setup=false` this is the job of the user
        @assert nfmu.currentComponent.state == fmi2ComponentStateInitializationMode "FMU needs to be in initialization mode after setup (start)."

        # find x0 (if not provided)
        if x_start === nothing 
            nfmu.x0 = fmi2GetContinuousStates(nfmu.currentComponent)
        else
            nfmu.x0 = x_start
            fmi2SetContinuousStates(nfmu.currentComponent, nfmu.x0)
        end

        # setup (hard) end
        if nfmu.fmu.executionConfig.setup
            fmi2ExitInitializationMode(nfmu.currentComponent)
        end

        # from here on, we are in event mode, if `setup=false` this is the job of the user
        @assert nfmu.currentComponent.state == fmi2ComponentStateEventMode "FMU needs to be in event mode after setup (end)."

        # this is not necessary for NeuralFMUs:
        #x0 = fmi2GetContinuousStates(nfmu.currentComponent)
        #x0_nom = fmi2GetNominalsOfContinuousStates(nfmu.currentComponent)

        # initial event handling
        handleEvents(nfmu.currentComponent) 
        fmi2EnterContinuousTimeMode(nfmu.currentComponent)

        nfmu.fmu.hasStateEvents = (nfmu.fmu.modelDescription.numberOfEventIndicators > 0)
        nfmu.fmu.hasTimeEvents = (nfmu.currentComponent.eventInfo.nextEventTimeDefined == fmi2True)

        if showProgress 
            progressMeter = ProgressMeter.Progress(1000; desc="Simulating ME-NeuralFMU ...", color=:black, dt=1.0) #, barglyphs=ProgressMeter.BarGlyphs("[=> ]"))
            ProgressMeter.update!(progressMeter, 0) # show it!
        end

        # custom callbacks
        for cb in nfmu.customCallbacks
            push!(callbacks, cb)
        end

        # integrator step callback
        if nfmu.fmu.hasStateEvents || nfmu.fmu.hasTimeEvents || showProgress
            stepCb = FunctionCallingCallback((x, t, integrator) -> stepCompleted(nfmu, x, t, integrator, progressMeter, t_start, t_stop);
                                                func_everystep=true,
                                                func_start=true)
            push!(callbacks, stepCb)
        end

        # state event callback
        handleStateEvents = nfmu.fmu.hasStateEvents && nfmu.fmu.executionConfig.handleStateEvents
        if handleStateEvents
            
            if nfmu.fmu.executionConfig.useVectorCallbacks

                eventCb = VectorContinuousCallback((out, x, t, integrator) -> condition(nfmu, out, x, t, integrator),
                                                (integrator, idx) -> affectFMU!(nfmu, integrator, idx),
                                                Int64(nfmu.fmu.modelDescription.numberOfEventIndicators);
                                                rootfind=RightRootFind,
                                                save_positions=(false, false),
                                                interp_points=nfmu.fmu.executionConfig.rootSearchInterpolationPoints) 
                push!(callbacks, eventCb)
            else

                for idx in 1:nfmu.fmu.modelDescription.numberOfEventIndicators
                    eventCb = ContinuousCallback((x, t, integrator) -> conditionSingle(nfmu, idx, x, t, integrator),
                                                    (integrator) -> affectFMU!(nfmu, integrator, idx);
                                                    rootfind=RightRootFind,
                                                    save_positions=(false, false),
                                                    interp_points=nfmu.fmu.executionConfig.rootSearchInterpolationPoints) 
                    push!(callbacks, eventCb)
                end
            end
        end

        handleTimeEvents = nfmu.fmu.hasTimeEvents && nfmu.fmu.executionConfig.handleTimeEvents
        if handleTimeEvents
            timeEventCb = IterativeCallback((integrator) -> time_choice(nfmu.fmu.components[end], integrator),
                                            (integrator) -> affectFMU!(nfmu, integrator, 0), 
                                            Float64; 
                                            initial_affect=(nfmu.fmu.components[end].eventInfo.nextEventTime == t_start), # true?
                                            save_positions=(false,false))
        
            push!(callbacks, timeEventCb)
        end
        
        @debug "NeuralFMU experimental event handling, stateEvents: $(handleStateEvents), timeEvents: $(handleTimeEvents)."

        # auto pick sensealg 
        if sense === nothing

            # currently, Callbacks only working with ForwardDiff
            if length(callbacks) > 0 
                p_len = 0 
                fp = Flux.params(nfmu)
                if length(fp) > 0
                    p_len = length(fp[1])
                end
                fds_chunk_size = p_len
                # limit to 256 because of RAM usage
                if fds_chunk_size > 256
                    fds_chunk_size = 256
                end
                
                sense = ForwardDiffSensitivity(;chunk_size=fds_chunk_size, convert_tspan=true) 
                #sense = QuadratureAdjoint(autojacvec=ZygoteVJP())
        
            else
                sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()) # EnzymeVJP()

                if inPlace === true
                    #@info "NeuralFMU: The configuration orders to use `inPlace=true`, but this is currently not supported by the (automatically) determined sensealg `Zygote`. Switching to `inPlace=false`. If you don't want this behaviour, explicitely choose a sensealg instead of `nothing`."
                    inPlace = false # Zygote only works for out-of-place (currently)
                end
            end
        end

        if saving 
            savedValues = SavedValues(Float64, Tuple{collect(Float64 for i in 1:length(nfmu.recordValues))...})

            savingCB = SavingCallback((x, t, integrator) -> saveValues(nfmu, nfmu.fmu.components[end], nfmu.recordValues, x, t, integrator), 
                              savedValues, 
                              saveat=nfmu.saveat)
            push!(callbacks, savingCB)
        end

    end # ignore_derivatives

    # setup problem and parameters
    p = nfmu.neuralODE.p
    prob = nothing

    if inPlace
        ff = ODEFunction{true}((dx, x, p, t) -> fx(nfmu, dx, x, p, t), 
                               tgrad=nothing)
        prob = ODEProblem{true}(ff, nfmu.x0, nfmu.tspan, p)
    else 
        ff = ODEFunction{false}((x, p, t) -> fx(nfmu, x, p, t), 
                                tgrad=basic_tgrad)
        prob = ODEProblem{false}(ff, nfmu.x0, nfmu.tspan, p)
    end

    if length(callbacks) > 0
        nfmu.solution = solve(prob, nfmu.neuralODE.args...; sensealg=sense, saveat=nfmu.saveat, callback=CallbackSet(callbacks...), nfmu.neuralODE.kwargs...) 
    else 
        nfmu.solution = solve(prob, nfmu.neuralODE.args...; sensealg=sense, saveat=nfmu.saveat, nfmu.neuralODE.kwargs...)
    end  

    # cleanup progress meter

    ignore_derivatives() do
        if showProgress 
            ProgressMeter.finish!(progressMeter)
        end

        if nfmu.currentComponent != nothing
            if nfmu.fmu.executionConfig.freeInstance
                fmi2FreeInstance!(nfmu.currentComponent)
            end
            nfmu.currentComponent = nothing
        end
    end # ignore_derivatives

    if saving
        return nfmu.solution, savedValues
    else 
        return nfmu.solution
    end
end

"""
Evaluates the CS_NeuralFMU in the timespan given during construction or in a custum timespan from `t_start` to `t_stop` with a given time step size `t_step`.

Via optional argument `reset`, the FMU is reset every time evaluation is started (default=`true`).
"""
function (nfmu::CS_NeuralFMU{F, C})(inputFct,
                                 t_step::Real, 
                                 t_start::Real = nfmu.tspan[1], 
                                 t_stop::Real = nfmu.tspan[end]; 
                                 reset::Bool = true,
                                 setup::Bool = true) where {F, C}

    ignore_derivatives() do
        if nfmu.fmu.executionConfig.instantiate

            nfmu.currentComponent = fmi2Instantiate!(nfmu.fmu)
            @debug "[NEW INST]"
        else
            nfmu.currentComponent = nfmu.fmu.components[end]
        end

        if nfmu.fmu.executionConfig.terminate
            retcode = fmi2Terminate(nfmu.currentComponent; soft=true)
            @assert retcode == fmi2StatusOK "CS_NeuralFMU(...): Terminating failed with return code $(retcode)."
        end
           
        if nfmu.fmu.executionConfig.reset
            retcode = fmi2Reset(nfmu.currentComponent; soft=true)
            @assert retcode == fmi2StatusOK "CS_NeuralFMU(...): Resetting failed with return code $(retcode)."
        end

        if nfmu.fmu.executionConfig.setup
            retcode = fmi2SetupExperiment(nfmu.currentComponent, t_start, t_stop) 
            @assert retcode == fmi2StatusOK "CS_NeuralFMU(...): Setting up experiment failed with return code $(retcode)."
            
            retcode = fmi2EnterInitializationMode(nfmu.currentComponent)
            @assert retcode == fmi2StatusOK "CS_NeuralFMU(...): Entering initialization mode failed with return code $(retcode)."
        end 

        if nfmu.fmu.executionConfig.setup
            retcode =  fmi2ExitInitializationMode(nfmu.currentComponent) 
            @assert retcode == fmi2StatusOK "CS_NeuralFMU(...): Exiting initialization mode failed with return code $(retcode)."
        end
    end # ignore_derivatives

    ts = t_start:t_step:t_stop
    nfmu.currentComponent.skipNextDoStep = true # skip first fim2DoStep-call
    model_input = inputFct.(ts)
    valueStack = nfmu.model.(model_input)

    ignore_derivatives() do
        if nfmu.currentComponent != nothing
            if nfmu.fmu.executionConfig.freeInstance
                fmi2FreeInstance!(nfmu.currentComponent)
            end
            nfmu.currentComponent = nothing
        end
    end # ignore_derivatives

    return valueStack

    # savedValues = SavedValues(Float64, Tuple{collect(Float64 for i in 1:length(nfmu.model(inputFct(t_start))))...})
    
    # ts = t_start:t_step:t_stop
    # i = 1
    # for t in ts
    #     DiffEqCallbacks.copyat_or_push!(savedValues.t, i, t)
    #     DiffEqCallbacks.copyat_or_push!(savedValues.saveval, i, (nfmu.model(inputFct(t))...,), Val{false})
    #     i += 1
    # end

    # return true, savedValues
end
function (nfmu::CS_NeuralFMU{Vector{F}, Vector{C}})(inputFct,
                                         t_step::Real, 
                                         t_start::Real = nfmu.tspan[1], 
                                         t_stop::Real = nfmu.tspan[end]; 
                                         reset::Bool = true,
                                         setup::Bool = true) where {F, C}
    
    ignore_derivatives() do
        for i in 1:length(nfmu.fmu)
            if nfmu.fmu[i].executionConfig.instantiate

                nfmu.currentComponent[i] = fmi2Instantiate!(nfmu.fmu[i])
                @debug "[NEW INST]"
            else
                nfmu.currentComponent[i] = nfmu.fmu[i].components[end]
            end

            if nfmu.fmu[i].executionConfig.terminate
                retcode = fmi2Terminate(nfmu.currentComponent[i]; soft=true)
                @assert retcode == fmi2StatusOK "CS_NeuralFMU(...): Terminating failed with return code $(retcode)."
            end
                
            if nfmu.fmu[i].executionConfig.reset
                retcode = fmi2Reset(nfmu.currentComponent[i]; soft=true)
                @assert retcode == fmi2StatusOK "CS_NeuralFMU(...): Resetting failed with return code $(retcode)."
            end

            if nfmu.fmu[i].executionConfig.setup
                retcode = fmi2SetupExperiment(nfmu.currentComponent[i], t_start, t_stop) 
                @assert retcode == fmi2StatusOK "CS_NeuralFMU(...): Setting up experiment failed with return code $(retcode)."
                
                retcode = fmi2EnterInitializationMode(nfmu.currentComponent[i])
                @assert retcode == fmi2StatusOK "CS_NeuralFMU(...): Entering initialization mode failed with return code $(retcode)."
            end 

            if nfmu.fmu[i].executionConfig.setup
                retcode =  fmi2ExitInitializationMode(nfmu.currentComponent[i]) 
                @assert retcode == fmi2StatusOK "CS_NeuralFMU(...): Exiting initialization mode failed with return code $(retcode)."
            end
        end
    end # ignore_derivatives

    ts = t_start:t_step:t_stop
    model_input = inputFct.(ts)
    valueStack = nfmu.model.(model_input)

    ignore_derivatives() do
        for i in 1:length(nfmu.fmu)
            if nfmu.currentComponent[i] != nothing
                if nfmu.fmu[i].executionConfig.freeInstance
                    fmi2FreeInstance!(nfmu.currentComponent[i])
                end
                nfmu.currentComponent[i] = nothing
            end
        end
    end # ignore_derivatives

    return valueStack

    # savedValues = SavedValues(Float64, Tuple{collect(Float64 for i in 1:length(recordValues))...})
    
    # ts = t_start:t_step:t_stop
    # i = 1
    # for t in ts
    #     DiffEqCallbacks.copyat_or_push!(savedValues.t, i, t)
    #     DiffEqCallbacks.copyat_or_push!(savedValues.saveval, i, (valueStack[i]...,), Val{false})
    #     i += 1
    # end

    # return true, savedValues
end

# adapting the Flux functions
function Flux.params(neuralfmu::ME_NeuralFMU)
    Flux.params(neuralfmu.neuralODE)
end

function Flux.params(neuralfmu::CS_NeuralFMU)
    Flux.params(neuralfmu.model)
end

# FMI version independent dosteps

"""
Wrapper. Call ```fmi2EvaluateME``` for more information.
"""
function fmiEvaluateME(str::fmi2Struct, 
                     x::Array{<:Real}, 
                     t::Real = (typeof(str) == FMU2 ? str.components[end].t : str.t), 
                     setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0), 
                     setValues::Array{<:Real} = zeros(Real, 0),
                     getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    fmi2EvaluateME(str, x, t,
                setValueReferences,
                setValues,
                getValueReferences)
end

"""
Wrapper. Call ```fmi2DoStepCS``` for more information.
"""
function fmiDoStepCS(str::fmi2Struct, 
                     dt::Real,
                     setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0), 
                     setValues::Array{<:Real} = zeros(Real, 0),
                     getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    fmi2DoStepCS(str, dt, setValueReferences, setValues, getValueReferences)
end

"""
Wrapper. Call ```fmi2InputDoStepCSOutput``` for more information.
"""
function fmiInputDoStepCSOutput(str::fmi2Struct, 
                                dt::Real, 
                                u::Array{<:Real})
    fmi2InputDoStepCSOutput(str, dt, u)
end

# define neutral gradients (=feed-trough) for ccall-functions (ToDo: Remove)
function neutralGradient(c̄)
    tuple(c̄,)
end

@adjoint fmi2SetupExperiment(fmu, startTime, stopTime) = fmi2SetupExperiment(fmu, startTime, stopTime), c̄ -> neutralGradient(c̄)
@adjoint fmi2EnterInitializationMode(fmu) = fmi2EnterInitializationMode(fmu), c̄ -> neutralGradient(c̄)
@adjoint fmi2ExitInitializationMode(fmu) = fmi2ExitInitializationMode(fmu), c̄ -> neutralGradient(c̄)
@adjoint fmi2Reset(fmu) = fmi2Reset(fmu), c̄ -> neutralGradient(c̄)
@adjoint fmi2Terminate(fmu) = fmi2Terminate(fmu), c̄ -> neutralGradient(c̄)

