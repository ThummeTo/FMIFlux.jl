#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Zygote
using Zygote: @adjoint

using Flux, DiffEqFlux
using OrdinaryDiffEq
using DiffEqCallbacks

using DiffEqFlux: ODEFunction, basic_tgrad, ODEProblem, ZygoteVJP, InterpolatingAdjoint, solve

import ForwardDiff
import Optim

include("FMI2_neural.jl")

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

    tspan
    saveat
    saved_values
    recordValues

    valueStack

    handleEvents::Bool

    ME_NeuralFMU() = new()
end

"""
Structure definition for a NeuralFMU, that runs in mode `Co-Simulation` (CS).
"""
mutable struct CS_NeuralFMU{T} <: NeuralFMU
    model
    #simulationResult::fmi2SimulationResult
    fmu::T

    tspan
    saveat
    valueStack
    
    CS_NeuralFMU{T}() where {T} = new{T}()
end

##### EVENT HANDLING START

# Read next time event from fmu and provide it to the integrator 
function time_choice(c::fmi2Component, integrator)
    eventInfo = fmi2NewDiscreteStates(c)
    fmi2EnterContinuousTimeMode(c)
    if Bool(eventInfo.nextEventTimeDefined)
        eventInfo.nextEventTime
    else
        Inf
    end
end

# Handles events and returns the values and nominals of the changed continuous states.
function handleEvents(c::fmi2Component, enterEventMode::Bool, exitInContinuousMode::Bool)

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
            @error "Event info returned error!"
        end
    end

    if exitInContinuousMode
        fmi2EnterContinuousTimeMode(c)
    end

    return valuesOfContinuousStatesChanged, nominalsOfContinuousStatesChanged
end

# Returns the event indicators for an FMU.
function condition(nfmu::ME_NeuralFMU, out, x, t, integrator) # Event when event_f(u,t) == 0

    if isa(t, ForwardDiff.Dual) 
        t = ForwardDiff.value(t)
    end 
    fmi2SetTime(nfmu.fmu, t)

    if all(isa.(x, ForwardDiff.Dual))
        x = collect(ForwardDiff.value(e) for e in x)
    end

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    nfmu.neuralODE.model(x) # evaluate NeuralFMU (set new states)

    indicators = fmi2GetEventIndicators(nfmu.fmu)
    #@info "Integrator stepped to t=$(t)s, indicators=$(indicators)"

    copy!(out, indicators)
end

function f_optim(x, nfmu, right_x_fmu, idx, direction::Real)
    # propagete the new state-guess `x` through the NeuralFMU
    nfmu.neuralODE.model(x)
    indicators = fmi2GetEventIndicators(nfmu.fmu)
    return Flux.Losses.mse(right_x_fmu, fmiGetContinuousStates(nfmu.fmu)) # - min(-direction*indicators[idx], 0.0)
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

    left_x_fmu = fmiGetContinuousStates(nfmu.fmu)

    continuousStatesChanged, nominalsChanged = handleEvents(nfmu.fmu.components[end], true, Bool(sign(idx)))

    #@info "Event [$idx] detected at t=$(t)s (statesChanged=$(continuousStatesChanged))"
    indicators = fmi2GetEventIndicators(nfmu.fmu)

    if continuousStatesChanged == fmi2True

        left_x = x

        right_x_fmu = fmiGetContinuousStates(nfmu.fmu) # the new FMU state after handled events

        #@info "NeuralFMU state event from $(left_x) (fmu: $(left_x_fmu)). Indicator [$idx]: $(indicators[idx])."

        # ToDo: Problem-related parameterization of optimize-call
        #result = optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu), left_x, LBFGS(); autodiff = :forward)
        result = Optim.optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu, idx, sign(indicators[idx])), left_x, NelderMead())

        #display(result)

        right_x = Optim.minimizer(result)
        integrator.u = right_x

        #indicators = fmi2GetEventIndicators(nfmu.fmu)
        #@info "NeuralFMU state event to   $(right_x) (fmu: $(right_x_fmu)). Indicator [$idx]: $(indicators[idx]). Minimum: $(Optim.minimum(result))."
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
    end
end

# save FMU values 
function saveValues(c::fmi2Component, recordValues, u, t, integrator)
    (fmiGetReal(c, recordValues)...,)
end

##### EVENT HANDLING END

"""
Constructs a ME-NeuralFMU where the FMU is at an arbitrary location inside of the NN.

Arguents:
    - `fmu` the considered FMU inside the NN 
    - `model` the NN topology (e.g. Flux.chain)
    - `tspan` simulation time span
    - `alg` a numerical ODE solver

Keyword arguments:
    - `saveat` time points to save the NeuralFMU output, if empty, solver step size is used (may be non-equidistant)
    - `fixstep` forces fixed step integration
    - `recordFMUValues` additionally records internal FMU variables (currently not supported because of open issues)
"""
function ME_NeuralFMU(fmu, 
                      model, 
                      tspan, 
                      alg=nothing; 
                      saveat=[], 
                      recordFMUValues = nothing, 
                      kwargs...)

    nfmu = ME_NeuralFMU()
    nfmu.fmu = fmu

    ext_model = nothing

    ext_model = Chain(model.layers...)

    nfmu.handleEvents = false
    nfmu.saved_values = nothing

    if (nfmu.fmu.modelDescription.numberOfEventIndicators > 0)
        @info "This ME-NeuralFMU has event indicators. Event-handling is in BETA-testing, so it is disabled by default. If you want to try event-handling during training for this discontinuous FMU, use the attribute `myNeuralFMU.handleEvents=true`."
    end

    nfmu.recordValues = prepareValueReference(fmu.components[end], recordFMUValues)

    nfmu.neuralODE = NeuralODE(ext_model, tspan, alg; saveat=saveat, kwargs...)

    nfmu.tspan = tspan
    nfmu.saveat = saveat

    nfmu
end

"""
Constructs a CS-NeuralFMU where the FMU is at an arbitrary location inside of the NN.

Arguents:
    - `fmu` the considered FMU inside the NN 
    - `model` the NN topology (e.g. Flux.chain)
    - `tspan` simulation time span

Keyword arguments:
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
Evaluates the ME_NeuralFMU in the timespan given during construction or in a custum timespan from `t_start` to `t_stop` for a given start state `x_start`.

Via optional argument `reset`, the FMU is reset every time evaluation is started (default=`true`).
Via optional argument `setup`, the FMU is set up every time evaluation is started (default=`true`).
Via optional argument `rootSearchInterpolationPoints`, the number of interpolation points during root search before event-handling is controlled (default=`100`). Try to increase this value, if event-handling fails with exception.
"""
function (nfmu::ME_NeuralFMU)(x_start::Array{<:Real}, 
                              t_start::Real = nfmu.tspan[1], 
                              t_stop = nfmu.tspan[end]; 
                              reset::Bool=true,
                              setup::Bool=true,
                              rootSearchInterpolationPoints::Integer=100,
                              kwargs...)

    if reset
        fmiReset(nfmu.fmu)
    end
    if setup
        fmiSetupExperiment(nfmu.fmu, t_start)
        fmiEnterInitializationMode(nfmu.fmu)
        fmiExitInitializationMode(nfmu.fmu)
    end

    x0 = x_start # [t_start, x_start...]

    c = nfmu.fmu.components[end]
    tspan = getfield(nfmu.neuralODE,:tspan)
    t_start = tspan[1]
    t_stop = tspan[end]

    #################

    callbacks = []
    sense = nothing
    savedValues = nothing

    saving = (length(nfmu.recordValues) > 0)

    eventHandling = (nfmu.fmu.modelDescription.numberOfEventIndicators > 0)
    handleTimeEvents = false

    Zygote.ignore() do
        if eventHandling
            eventInfo = fmi2NewDiscreteStates(c)
            handleTimeEvents = (eventInfo.nextEventTimeDefined == fmi2True)
        end

        # First evaluation of the FMU
        x0 = fmi2GetContinuousStates(c)
        x0_nom = fmi2GetNominalsOfContinuousStates(c)

        fmi2SetContinuousStates(c, x0)
        
        handleEvents(c, false, false)

        # Get states of handling initial Events
        x0 = fmi2GetContinuousStates(c)
        x0_nom = fmi2GetNominalsOfContinuousStates(c)

        fmi2EnterContinuousTimeMode(c)
    end

    Zygote.ignore() do
        # Event handling only if there are event-indicators
        if nfmu.handleEvents && eventHandling

            # Only ForwardDiffSensitivity supports differentiation and callbacks
            
            #sense = ReverseDiffAdjoint() 
            sense = ForwardDiffSensitivity(;chunk_size=0,convert_tspan=true) 

            eventCb = VectorContinuousCallback((out, x, t, integrator) -> condition(nfmu, out, x, t, integrator),
                                            (integrator, idx) -> affectFMU!(nfmu, integrator, idx),
                                            Int64(nfmu.fmu.modelDescription.numberOfEventIndicators);
                                            rootfind=DiffEqBase.RightRootFind,
                                            save_positions=(false, false),
                                            interp_points=rootSearchInterpolationPoints) 
            push!(callbacks, eventCb)

            stepCb = FunctionCallingCallback((x, t, integrator) -> stepCompleted(nfmu, x, t, integrator);
                                            func_everystep=true,
                                            func_start=true)
            push!(callbacks, stepCb)

            if handleTimeEvents
                timeEventCb = IterativeCallback((integrator) -> time_choice(c, integrator),
                                                (integrator) -> affectFMU!(nfmu, integrator, 0), Float64; 
                                                initial_affect = true,
                                                save_positions=(false,false))
            
                push!(callbacks, timeEventCb)
            end
        else
            sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
        end

        if saving 
            savedValues = SavedValues(Float64, Tuple{collect(Float64 for i in 1:length(nfmu.recordValues))...})

            savingCB = SavingCallback((u,t,integrator) -> saveValues(c, nfmu.recordValues, u, t, integrator), 
                              savedValues, 
                              saveat=nfmu.saveat)
            push!(callbacks, savingCB)
        end
    end

    p = nfmu.neuralODE.p
    dudt_(u,p,t) = nfmu.neuralODE.re(p)(u)
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,x0,tspan,p)

    #sense = ForwardDiffSensitivity() # (;chunk_size=0,convert_tspan=true) 
    #callbacks = []

    # Zygote.ignore() do
    #     @info "Tspan: $(tspan)"
    #     @info "Sensealg: $(sense)"
    #     @info "Callbacks: $(length(callbacks))"
    # end
    
    if length(callbacks) > 0
        nfmu.solution = solve(prob,nfmu.neuralODE.args...;sensealg=sense, saveat=nfmu.saveat, callback=CallbackSet(callbacks...), nfmu.neuralODE.kwargs...)  
    else
        nfmu.solution = solve(prob,nfmu.neuralODE.args...;sensealg=sense, saveat=nfmu.saveat, nfmu.neuralODE.kwargs...)  
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

    if reset
        while fmiReset(nfmu.fmu) != 0
        end
    end 

    if setup
        while fmiSetupExperiment(nfmu.fmu, t_start) != 0
        end
        while fmiEnterInitializationMode(nfmu.fmu) != 0
        end
        while fmiExitInitializationMode(nfmu.fmu) != 0
        end
    end

    ts = t_start:t_step:t_stop
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
            while fmiReset(fmu) != 0
            end
        end
    end

    if setup
        for fmu in nfmu.fmu
            while fmiSetupExperiment(fmu, t_start) != 0
            end
            while fmiEnterInitializationMode(fmu) != 0
            end
            while fmiExitInitializationMode(fmu) != 0
            end
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
Wrapper. Call ```fmi2DoStepME``` for more information.
"""
function fmiDoStepME(fmu::FMU2, 
                     x::Array{<:Real}, 
                     t::Real = -1.0, 
                     setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0), 
                     setValues::Array{<:Real} = zeros(Real, 0),
                     getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    fmi2DoStepME(fmu, x, t,
                setValueReferences,
                setValues,
                getValueReferences)
end

"""
Wrapper. Call ```fmi2DoStepCS``` for more information.
"""
function fmiDoStepCS(fmu::FMU2, 
                     dt::Real,
                     setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0), 
                     setValues::Array{<:Real} = zeros(Real, 0),
                     getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    fmi2DoStepCS(fmu, dt, setValueReferences, setValues = setValues, getValueReferences)
end

"""
Wrapper. Call ```fmi2InputDoStepCSOutput``` for more information.
"""
function fmiInputDoStepCSOutput(fmu::FMU2, 
                                dt::Real, 
                                u::Array{<:Real})
    fmi2InputDoStepCSOutput(fmu, dt, u)
end

# define neutral gradients (=feed-trough) for ccall-functions (ToDo: Remove)
function neutralGradient(c̄)
    tuple(c̄,)
end

@adjoint fmiSetupExperiment(fmu, startTime, stopTime) = fmiSetupExperiment(fmu, startTime, stopTime), c̄ -> neutralGradient(c̄)
@adjoint fmiEnterInitializationMode(fmu) = fmiEnterInitializationMode(fmu), c̄ -> neutralGradient(c̄)
@adjoint fmiExitInitializationMode(fmu) = fmiExitInitializationMode(fmu), c̄ -> neutralGradient(c̄)
@adjoint fmiReset(fmu) = fmiReset(fmu), c̄ -> neutralGradient(c̄)

