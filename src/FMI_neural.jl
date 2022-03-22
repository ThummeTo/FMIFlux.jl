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

import SciMLBase: RightRootFind

using FMIImport: fmi2ComponentStateModelInitialized, fmi2ComponentStateModelSetableFMUstate, fmi2ComponentStateModelUnderEvaluation
import ChainRulesCore: ignore_derivatives

"""
The mutable struct representing an abstract (simulation mode unknown) NeuralFMU.
"""
abstract type NeuralFMU end

mutable struct NeuralFMU_TrainingModeConfig 
    terminate::Bool     # call fmi2Terminate before every training step
    reset::Bool         # call fmi2Reset before every training step
    instantiate::Bool   # call fmi2Instantiate before every training step
    freeInstance::Bool  # call fmi2FreeInstance after every training step

    handleStateEvents::Bool 
    handleTimeEvents::Bool 

    sensealg                                # algorithm for sensitivity estimation
    useCachedDersSense::Bool                # whether ẋ should be cached for frule/rrule (useful for ForwardDiff)
    rootSearchInterpolationPoints::UInt     # number of root search interpolation points
    inPlace::Bool                           # whether faster in-place-fx should be used
    useVectorCallbacks::Bool                # whether to use vector or scalar callbacks

    function NeuralFMU_TrainingModeConfig()
        inst = new()

        inst.terminate = true 
        inst.reset = true
        inst.instantiate = false 
        inst.freeInstance = false

        inst.handleStateEvents = false
        inst.handleTimeEvents = false

        inst.sensealg = nothing # auto
        inst.useCachedDersSense = true
        inst.rootSearchInterpolationPoints = 100
        inst.inPlace = true
        inst.useVectorCallbacks = true

        return inst 
    end
end

# default for a "healthy" FMU
DEFAULT_TRAINING_MODE_CONFIG = NeuralFMU_TrainingModeConfig()

# if your FMU has a problem with "fmi2Reset"
NO_RESET_TRAINING_MODE_CONFIG = NeuralFMU_TrainingModeConfig() 
NO_RESET_TRAINING_MODE_CONFIG.terminate = false
NO_RESET_TRAINING_MODE_CONFIG.reset = false
NO_RESET_TRAINING_MODE_CONFIG.instantiate = true
NO_RESET_TRAINING_MODE_CONFIG.freeInstance = true

# if your FMU has a problem with "fmi2FreeInstance"
NO_FREEING_TRAINING_MODE_CONFIG = NeuralFMU_TrainingModeConfig() 
NO_FREEING_TRAINING_MODE_CONFIG.terminate = false
NO_FREEING_TRAINING_MODE_CONFIG.reset = false
NO_FREEING_TRAINING_MODE_CONFIG.instantiate = true
NO_FREEING_TRAINING_MODE_CONFIG.freeInstance = false

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

    trainingConfig::NeuralFMU_TrainingModeConfig

    function ME_NeuralFMU()
        inst = new()
        inst.currentComponent = nothing

        inst.trainingConfig = DEFAULT_TRAINING_MODE_CONFIG

        return inst 
    end
end

"""
Structure definition for a NeuralFMU, that runs in mode `Co-Simulation` (CS).
"""
mutable struct CS_NeuralFMU{T} <: NeuralFMU
    model
    fmu::T

    tspan
    saveat
    valueStack

    trainingConfig::NeuralFMU_TrainingModeConfig
    
    CS_NeuralFMU{T}() where {T} = new{T}()
end

##### EVENT HANDLING START

# Read next time event from fmu and provide it to the integrator 
function time_choice(c::FMU2Component, integrator)
    eventInfo = fmi2NewDiscreteStates(c)
    fmi2EnterContinuousTimeMode(c)
    if eventInfo.nextEventTimeDefined == fmi2True
        ignore_derivatives() do 
            @debug "time_choice(_, _): Next event defined at $(eventInfo.nextEventTime)s"
        end 
        return eventInfo.nextEventTime
    end

    return nothing
end

# Handles events and returns the values and nominals of the changed continuous states.
function handleEvents(c::FMU2Component, enterEventMode::Bool, exitInContinuousMode::Bool)

    if enterEventMode
        fmi2EnterEventMode(c)
    end

    eventInfo = fmi2NewDiscreteStates(c)

    valuesOfContinuousStatesChanged = eventInfo.valuesOfContinuousStatesChanged
    nominalsOfContinuousStatesChanged = eventInfo.nominalsOfContinuousStatesChanged

    #set inputs here
    #fmiSetReal(myFMU, InputRef, Value)

    while eventInfo.newDiscreteStatesNeeded == fmi2True
        # update discrete states
        eventInfo = fmi2NewDiscreteStates(c)
        valuesOfContinuousStatesChanged = eventInfo.valuesOfContinuousStatesChanged
        nominalsOfContinuousStatesChanged = eventInfo.nominalsOfContinuousStatesChanged

        if eventInfo.terminateSimulation == fmi2True
            ignore_derivatives() do 
                @error "Event info returned error!"
            end
        end
    end

    if exitInContinuousMode
        fmi2EnterContinuousTimeMode(c)
    end

    return valuesOfContinuousStatesChanged, nominalsOfContinuousStatesChanged
end

# Returns the event indicators for an FMU.
function condition(nfmu::ME_NeuralFMU, out, x, t, integrator) # Event when event_f(u,t) == 0

    # ToDo: set inputs here
    #fmiSetReal(myFMU, InputRef, Value)

    if isa(t, ForwardDiff.Dual) 
        t = ForwardDiff.value(t)
    end 

    if all(isa.(x, ForwardDiff.Dual))
        x = collect(ForwardDiff.value(e) for e in x)
    end

    fmi2SetTime(nfmu.fmu, t)
    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    nfmu.neuralODE.model(x) # evaluate NeuralFMU (set new states)

    indicators = fmi2GetEventIndicators(nfmu.fmu)
    ignore_derivatives() do 
        @debug "condition(_, _, $(x), $(t), _): Event indicators=$(indicators)"
    end

    copy!(out, indicators)
end
function conditionSingle(nfmu::ME_NeuralFMU, index, x, t, integrator) 

    # ToDo: set inputs here
    #fmiSetReal(myFMU, InputRef, Value)

    if isa(t, ForwardDiff.Dual) 
        t = ForwardDiff.value(t)
    end 

    if all(isa.(x, ForwardDiff.Dual))
        x = collect(ForwardDiff.value(e) for e in x)
    end

    fmi2SetTime(nfmu.fmu, t)
    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    nfmu.neuralODE.model(x) # evaluate NeuralFMU (set new states)

    indicators = fmi2GetEventIndicators(nfmu.fmu)
    ignore_derivatives() do 
        @debug "condition(_, _, $(x), $(t), _): Event indicators=$(indicators)"
    end

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
    # Event found - handle it

    t = integrator.t
    if isa(t, ForwardDiff.Dual) 
        t = ForwardDiff.value(t)
    end 

    x = integrator.u
    if all(isa.(x, ForwardDiff.Dual))
        x = collect(ForwardDiff.value(e) for e in x)
    end

    left_x_fmu = fmi2GetContinuousStates(nfmu.fmu)

    continuousStatesChanged, nominalsChanged = handleEvents(nfmu.fmu.components[end], true, Bool(sign(idx)))

    ignore_derivatives() do  
        @debug "affectFMU!(_, _, $idx): Event [$idx] detected at t=$(t)s (statesChanged=$(continuousStatesChanged))"
    end
    #indicators = fmi2GetEventIndicators(nfmu.fmu)

    if continuousStatesChanged == fmi2True

        left_x = x

        right_x_fmu = fmi2GetContinuousStates(nfmu.fmu) # the new FMU state after handled events

        ignore_derivatives() do 
            @debug "affectFMU!(_, _, $idx): NeuralFMU state event from $(left_x) (fmu: $(left_x_fmu)). Indicator [$idx]: $(indicators[idx]). Optimizing new state ..."
        end

        # ToDo: Problem-related parameterization of optimize-call
        #result = optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu), left_x, LBFGS(); autodiff = :forward)
        #result = Optim.optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu, idx, sign(indicators[idx])), left_x, NelderMead())
        result = Optim.optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu), left_x, NelderMead())

        #display(result)

        right_x = Optim.minimizer(result)
        integrator.u = right_x

        ignore_derivatives() do 
            @debug "affectFMU!(_, _, $idx): NeuralFMU state event to   $(right_x) (fmu: $(right_x_fmu)). Indicator [$idx]: $(fmi2GetEventIndicators(nfmu.fmu)[idx]). Minimum: $(Optim.minimum(result))."
        end
    end

    if nominalsChanged == fmi2True
        x_nom = fmi2GetNominalsOfContinuousStates(nfmu.fmu)
    end
end

# Does one step in the simulation.
function stepCompleted(nfmu::ME_NeuralFMU, x, t, integrator)
    (status, enterEventMode, terminateSimulation) = fmi2CompletedIntegratorStep(nfmu.fmu, fmi2True)
    if enterEventMode == fmi2True
        affectFMU!(nfmu, integrator, 0)
    else
        # ToDo: set inputs here
        #fmiSetReal(myFMU, InputRef, Value)
    end

    ignore_derivatives() do
        
        if isa(t, ForwardDiff.Dual) 
            t = ForwardDiff.value(t)
        end 

        if t == nfmu.tspan[1] # FIRST STEP 
            @debug "[FIRST STEP]"

            if nfmu.trainingConfig.instantiate
                nfmu.currentComponent = fmi2Instantiate!(nfmu.fmu)
                @debug "[NEW INST]"
            else
                nfmu.currentComponent = nfmu.fmu.components[end]
            end

            if nfmu.trainingConfig.terminate
                if nfmu.currentComponent.state == fmi2ComponentStateModelInitialized
                    fmi2Terminate(nfmu.currentComponent)
                end
            end

            if nfmu.trainingConfig.reset
                if nfmu.currentComponent.state == fmi2ComponentStateModelSetableFMUstate
                    fmi2Reset(nfmu.currentComponent)
                end
            end
            
            fmi2SetupExperiment(nfmu.currentComponent, nfmu.tspan[1], nfmu.tspan[end])
            fmi2EnterInitializationMode(nfmu.currentComponent)
            fmi2SetContinuousStates(nfmu.currentComponent, nfmu.x0)
            fmi2ExitInitializationMode(nfmu.currentComponent)
        end
        
        if t == nfmu.tspan[end]
            @debug "[LAST STEP]"

            if nfmu.currentComponent != nothing
                if nfmu.trainingConfig.freeInstance
                    fmi2FreeInstance!(nfmu.currentComponent)
                end

                nfmu.currentComponent = nothing
            end

        end
    end
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

function fd_eltypes(e::Array{<:ForwardDiff.Dual{T, V, N}}) where {T, V, N}
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
        if nfmu.trainingConfig.useCachedDersSense
            if nfmu.firstRun
                if t <= nfmu.tspan[end]
                    if length(nfmu.fmu.t_cache) == 0 || nfmu.fmu.t_cache[end] < t
                        push!(nfmu.fmu.ẋ_cache, dx)
                        push!(nfmu.fmu.t_cache, t)
                    end
                end

                if t >= nfmu.tspan[end]
                    nfmu.firstRun = false 
                    nfmu.fmu.ẋ_interp = LinearInterpolation(nfmu.fmu.t_cache, nfmu.fmu.ẋ_cache)
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

    ext_model = nothing

    ext_model = Chain(model.layers...)

    nfmu.saved_values = nothing

    if (nfmu.fmu.modelDescription.numberOfEventIndicators > 0)
        @info "This ME-NeuralFMU has event indicators. Event-handling is in BETA-testing, so it is disabled by default. If you want to try event-handling during training for this discontinuous FMU, use the attributes `myNeuralFMU.config.handleStateEvents=true` and `myNeuralFMU.config.handleTimeEvents=true`."
    end

    nfmu.recordValues = prepareValueReference(fmu, recordFMUValues)

    nfmu.neuralODE = NeuralODE(ext_model, tspan, alg; saveat=saveat, kwargs...)

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
function CS_NeuralFMU(fmu, 
                      model, 
                      tspan; 
                      saveat=[], 
                      recordValues = [])

    nfmu = CS_NeuralFMU{typeof(fmu)}()

    nfmu.fmu = fmu

    nfmu.model = Chain(model.layers...)

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
                              t_start::Real = nfmu.tspan[1], 
                              t_stop = nfmu.tspan[end]; 
                              reset::Bool=true,
                              setup::Bool=true,
                              kwargs...)
    ignore_derivatives() do
        @debug "ME_NeuralFMU..."
    end

    saving = (length(nfmu.recordValues) > 0)
    handleStateEvents = nfmu.trainingConfig.handleStateEvents && (nfmu.fmu.modelDescription.numberOfEventIndicators > 0)
    handleTimeEvents = nfmu.trainingConfig.handleTimeEvents

    nfmu.firstRun = true
    nfmu.fmu.t_cache = [] 
    nfmu.fmu.ẋ_cache = []

    ########

    callbacks = []
    sense = nfmu.trainingConfig.sensealg
    inPlace = nfmu.trainingConfig.inPlace

    tspan = getfield(nfmu.neuralODE,:tspan)
    t_start = tspan[1]
    t_stop = tspan[end]
    
    if x_start === nothing 
        nfmu.x0 = fmi2GetContinuousStates(nfmu.fmu)
    else
        nfmu.x0 = x_start
    end

    #################

    ignore_derivatives() do

        # check for time events 
        eventInfo = fmi2NewDiscreteStates(nfmu.fmu.components[end])
        if handleTimeEvents
            handleTimeEvents = (eventInfo.nextEventTimeDefined == fmi2True)
        end

        # custom callbacks
        for cb in nfmu.customCallbacks
            push!(callbacks, cb)
        end

        # integrator step callback
        stepCb = FunctionCallingCallback((x, t, integrator) -> stepCompleted(nfmu, x, t, integrator);
                                            func_everystep=true,
                                            func_start=true)
        push!(callbacks, stepCb)

        # state event callback
        if handleStateEvents
            @debug "NeuralFMU experimental event handling, stateEvents: $(handleStateEvents), timeEvents: $(handleTimeEvents)."

            if nfmu.trainingConfig.useVectorCallbacks

                eventCb = VectorContinuousCallback((out, x, t, integrator) -> condition(nfmu, out, x, t, integrator),
                                                (integrator, idx) -> affectFMU!(nfmu, integrator, idx),
                                                Int64(nfmu.fmu.modelDescription.numberOfEventIndicators);
                                                rootfind=RightRootFind,
                                                save_positions=(false, false),
                                                interp_points=nfmu.trainingConfig.rootSearchInterpolationPoints) 
                push!(callbacks, eventCb)
            else

                for idx in 1:nfmu.fmu.modelDescription.numberOfEventIndicators
                    eventCb = ContinuousCallback((x, t, integrator) -> conditionSingle(nfmu, idx, x, t, integrator),
                                                    (integrator) -> affectFMU!(nfmu, integrator, idx);
                                                    rootfind=RightRootFind,
                                                    save_positions=(false, false),
                                                    interp_points=nfmu.trainingConfig.rootSearchInterpolationPoints) 
                    push!(callbacks, eventCb)
                end
            end

            if handleTimeEvents
                timeEventCb = IterativeCallback((integrator) -> time_choice(nfmu.fmu.components[end], integrator),
                                                (integrator) -> affectFMU!(nfmu, integrator, 0), 
                                                Float64; 
                                                initial_affect=false, # true?
                                                save_positions=(false,false))
            
                push!(callbacks, timeEventCb)
            end
        end

        # auto pick sensealg 
        if sense === nothing

            # currently, Callbacks only working with ForwardDiff
            if length(callbacks) > 0 
                p_len = length(Flux.params(nfmu)[1])
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
                    @warn "NeuralFMU: The configuration orders to use `inPlace=true`, but this is currently not supported by the (automatically) determined sensealg `Zygote`. Switching to `inPlace=false`. If you don't want this behaviour, explicitely choose a sensealg instead of `nothing`."
                    inPlace = false # Zygote only works for out-of-place
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
    end

    # setup problem and parameters
    p = nfmu.neuralODE.p
    prob = nothing

    if inPlace
        ff = ODEFunction{true}((dx, x, p, t) -> fx(nfmu, dx, x, p, t), 
                               tgrad=nothing)
        prob = ODEProblem{true}(ff, nfmu.x0, tspan, p)
    else 
        ff = ODEFunction{false}((x, p, t) -> fx(nfmu, x, p, t), 
                                tgrad=basic_tgrad)
        prob = ODEProblem{false}(ff, nfmu.x0, tspan, p)
    end

    if length(callbacks) > 0
        nfmu.solution = solve(prob, nfmu.neuralODE.args...; sensealg=sense, saveat=nfmu.saveat, callback=CallbackSet(callbacks...), nfmu.neuralODE.kwargs...) 
    else 
        nfmu.solution = solve(prob, nfmu.neuralODE.args...; sensealg=sense, saveat=nfmu.saveat, nfmu.neuralODE.kwargs...)
    end  

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
function (nfmu::CS_NeuralFMU{T})(inputFct,
                                 t_step::Real, 
                                 t_start::Real = nfmu.tspan[1], 
                                 t_stop::Real = nfmu.tspan[end]; 
                                 reset::Bool = true,
                                 setup::Bool = true) where {T}

    c = nfmu.fmu.components[end]
    
    if reset
        if c.state == fmi2ComponentStateModelInitialized 
            fmi2Terminate(c)
        end
        if c.state == fmi2ComponentStateModelSetableFMUstate
            fmi2Reset(c)
        end
    end

    if setup
        while fmi2SetupExperiment(nfmu.fmu, t_start, t_stop) != 0
        end
        while fmi2EnterInitializationMode(nfmu.fmu) != 0
        end
        while fmi2ExitInitializationMode(nfmu.fmu) != 0
        end
    end

    ts = t_start:t_step:t_stop
    nfmu.fmu.components[end].skipNextDoStep = true # skip first fim2DoStep-call
    model_input = inputFct.(ts)
    valueStack = nfmu.model.(model_input)
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
function (nfmu::CS_NeuralFMU{Vector{T}})(inputFct,
                                         t_step::Real, 
                                         t_start::Real = nfmu.tspan[1], 
                                         t_stop::Real = nfmu.tspan[end]; 
                                         reset::Bool = true,
                                         setup::Bool = true) where {T}
    
    if reset
        for fmu in nfmu.fmu 
            c = fmu.components[end]
            if c.state == fmi2ComponentStateModelInitialized
                fmi2Terminate(c)
            end
            if c.state == fmi2ComponentStateModelSetableFMUstate
                fmi2Reset(c)
            end
        end 
    end

    if setup
        for fmu in nfmu.fmu
            while fmi2SetupExperiment(fmu, t_start, t_stop) != 0
            end
            while fmi2EnterInitializationMode(fmu) != 0
            end
            while fmi2ExitInitializationMode(fmu) != 0
            end

            fmu.components[end].skipNextDoStep = true # skip first fim2DoStep-call
        end
    end

    ts = t_start:t_step:t_stop
    model_input = inputFct.(ts)
    valueStack = nfmu.model.(model_input)
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
                     t::Real = -1.0, 
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

