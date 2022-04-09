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

using FMIImport: fmi2StatusOK, FMU2Solution, FMU2Event

"""
The mutable struct representing an abstract (simulation mode unknown) NeuralFMU.
"""
abstract type NeuralFMU end

"""
Structure definition for a NeuralFMU, that runs in mode `Model Exchange` (ME).
"""
mutable struct ME_NeuralFMU <: NeuralFMU
    neuralODE::NeuralODE
    fmu::FMU

    currentComponent

    tspan
    saveat
    saved_values
    recordValues

    valueStack

    customCallbacks::Array
    callbacks::Array

    x0::Array{Float64}
    firstRun::Bool
    solution::FMU2Solution

    tolerance::Union{Real, Nothing}
    parameters::Union{Dict{<:Any, <:Any}, Nothing} 
    setup::Union{Bool, Nothing} 
    reset::Union{Bool, Nothing}
    instantiate::Union{Bool, Nothing} 
    freeInstance::Union{Bool, Nothing} 
    terminate::Union{Bool, Nothing} 

    progressMeter

    #componentShadow::Union{FMU2ComponentShadow, Nothing}

    solveCycle::UInt
   
    function ME_NeuralFMU()
        inst = new()
        inst.currentComponent = nothing
        inst.progressMeter = nothing

        return inst 
    end
end

"""
Structure definition for a NeuralFMU, that runs in mode `Co-Simulation` (CS).
"""
mutable struct CS_NeuralFMU{F, C} <: NeuralFMU
    model
    fmu::F
    currentComponent # ::Union{C, Nothing}

    tspan
    saveat
    solution::FMU2Solution

    function CS_NeuralFMU{F, C}() where {F, C} 
        inst = new{F, C}()

        inst.currentComponent = nothing

        return inst
    end
end

##### EVENT HANDLING START

function startCallback(nfmu, t)
    ignore_derivatives() do
        nfmu.solveCycle += 1
        @debug "[$(nfmu.solveCycle)][FIRST STEP]"
        
        @assert ForwardDiff.value(t) == nfmu.tspan[1] "startCallback(...): Called for non-start-point t=$(ForwardDiff.value(t))"
        
        # allocate component shadow for second run
        if nfmu.fmu.executionConfig.useComponentShadow

            if nfmu.currentComponent == nothing
                nfmu.currentComponent = FMU2ComponentShadow()
                @debug "[$(nfmu.solveCycle)][INIT SHADOW]"
                push!(nfmu.fmu.components, nfmu.currentComponent)
            end

            # real comp renew
            realComponent = nfmu.currentComponent.realComponent
            realComponent, nfmu.x0 = prepareFMU(nfmu.fmu, realComponent, nfmu.instantiate, nfmu.terminate, nfmu.reset, nfmu.setup, nfmu.parameters, nfmu.tspan[1], nfmu.tspan[end], nfmu.tolerance; x0=nfmu.x0, pushComponents=false)
            setComponent!(nfmu.currentComponent, realComponent)

            handleEvents(nfmu.currentComponent) 
            fmi2EnterContinuousTimeMode(nfmu.currentComponent)

            nfmu.currentComponent.log = (nfmu.solveCycle == 2)

        else
            nfmu.currentComponent, nfmu.x0 = prepareFMU(nfmu.fmu, nfmu.currentComponent, nfmu.instantiate, nfmu.terminate, nfmu.reset, nfmu.setup, nfmu.parameters, nfmu.tspan[1], nfmu.tspan[end], nfmu.tolerance; x0=nfmu.x0)

            handleEvents(nfmu.currentComponent) 
            fmi2EnterContinuousTimeMode(nfmu.currentComponent)
        end

        # re-init every callback except #1 (the startCallback itself)
        # for i in 2:length(nfmu.callbacks)
        #     callback = nfmu.callbacks[i]
        #     callback.initialize(callback, u, t, integrator)
        # end
    end
end

function stopCallback(nfmu, t)
    ignore_derivatives() do
        @debug "[$(nfmu.solveCycle)][LAST STEP]"

        @assert ForwardDiff.value(t) == nfmu.tspan[end] "stopCallback(...): Called for non-start-point t=$(ForwardDiff.value(t))"
        
        if nfmu.fmu.executionConfig.useComponentShadow

            if nfmu.solveCycle == 2 
                prepare!(nfmu.currentComponent)
                @debug "[$(nfmu.solveCycle)][PREPARED SHADOW]"
            end
            
            comp = finishFMU(nfmu.fmu, nfmu.currentComponent.realComponent, nfmu.freeInstance; popComponent=false)
            setComponent!(nfmu.currentComponent, comp)
        else
            nfmu.currentComponent = finishFMU(nfmu.fmu, nfmu.currentComponent, nfmu.freeInstance)
        end
        
    end
end

# Read next time event from fmu and provide it to the integrator 
function time_choice(nfmu, integrator)

    # last call may be after simulation end
    if nfmu.currentComponent == nothing 
        return nothing 
    end

    if nfmu.currentComponent.eventInfo.nextEventTimeDefined == fmi2True
        #@debug "time_choice(...): $(nfmu.currentComponent.eventInfo.nextEventTime) at t=$(ForwardDiff.value(integrator.t))"
        return nfmu.currentComponent.eventInfo.nextEventTime
    else
        #@debug "time_choice(...): nothing at t=$(ForwardDiff.value(integrator.t))"
        return nothing
    end
end

# Handles events and returns the values and nominals of the changed continuous states.
function handleEvents(c::Union{FMU2Component, FMU2ComponentShadow})

    @assert c.state == fmi2ComponentStateEventMode "handleEvents(...): Must be in event mode!"

    #@debug "Handle Events..."

    # trigger the loop
    c.eventInfo.newDiscreteStatesNeeded = fmi2True

    valuesOfContinuousStatesChanged = fmi2False
    nominalsOfContinuousStatesChanged = fmi2False
    nextEventTimeDefined = fmi2False
    nextEventTime = 0.0

    numCalls = 0
    while c.eventInfo.newDiscreteStatesNeeded == fmi2True
        numCalls += 1
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

        @assert numCalls <= c.fmu.executionConfig.maxNewDiscreteStateCalls "handleEvents(...): `fmi2NewDiscreteStates!` exceeded $(c.fmu.executionConfig.maxNewDiscreteStateCalls) calls, this may be an error in the FMU. If not, you can change the max value for this FMU in `fmu.executionConfig.maxNewDiscreteStateCalls`."
    end

    c.eventInfo.valuesOfContinuousStatesChanged = valuesOfContinuousStatesChanged
    c.eventInfo.nominalsOfContinuousStatesChanged = nominalsOfContinuousStatesChanged
    c.eventInfo.nextEventTimeDefined = nextEventTimeDefined
    c.eventInfo.nextEventTime = nextEventTime

    return nothing
end

# Returns the event indicators for an FMU.
function condition(nfmu::ME_NeuralFMU, out::SubArray{<:ForwardDiff.Dual{T, V, N}, A, B, C, D}, x, t, integrator) where {T, V, N, A, B, C, D} # Event when event_f(u,t) == 0

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

    buf = zeros(nfmu.fmu.modelDescription.numberOfEventIndicators)
    fmi2GetEventIndicators!(nfmu.fmu.components[end], buf)

    out[:] = collect(ForwardDiff.Dual{T, V, N}(V(buf[i]), ForwardDiff.partials(out[i])) for i in 1:length(out))
    
    return nothing
end
function condition(nfmu::ME_NeuralFMU, out, x, t, integrator) # Event when event_f(u,t) == 0

    # last call may be after simulation end
    # if nfmu.currentComponent == nothing 
    #     return nothing 
    # end

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

    #@debug "Condition..."

    fmi2GetEventIndicators!(nfmu.fmu.components[end], out)
    
    return nothing
end
function conditionSingle(nfmu::ME_NeuralFMU, index, x, t, integrator) 

    # last call may be after simulation end
    # if nfmu.currentComponent == nothing 
    #     return 1.0
    # end

    @assert nfmu.fmu.components[end].state == fmi2ComponentStateContinuousTimeMode "condition(...): Must be called in mode continuous time."

    # ToDo: set inputs here
    #fmiSetReal(myFMU, InputRef, Value)

    if isa(t, ForwardDiff.Dual) 
        t = ForwardDiff.value(t)
    end 

    if all(isa.(x, ForwardDiff.Dual))
        x = collect(ForwardDiff.value(e) for e in x)
    end

    #@debug "ConditionSingle..."

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    nfmu.neuralODE.model(x) # evaluate NeuralFMU (set new states)
    fmi2SetTime(nfmu.fmu, t)

    # ToDo: Input Function

    indicators = zeros(nfmu.fmu.modelDescription.numberOfEventIndicators)
    fmi2GetEventIndicators!(nfmu.fmu, indicators)

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

    # ignore_derivatives() do 
    #     if idx == 0 
    #         @debug "affectFMU!(...): Handle time event at t=$t"
    #     end

    #     if idx > 0 
    #         @debug "affectFMU!(...): Handle state event at t=$t"
    #     end
    # end 

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
        result = Optim.optimize(x_seek -> f_optim(x_seek, nfmu, right_x_fmu), left_x, NelderMead())

        #display(result)

        right_x = Optim.minimizer(result)
        integrator.u = right_x

    end

    if c.eventInfo.nominalsOfContinuousStatesChanged == fmi2True
        x_nom = fmi2GetNominalsOfContinuousStates(nfmu.fmu)
    end

    ignore_derivatives() do 
        if idx != -1
            e = FMU2Event(t, UInt64(idx), left_x, right_x)
            push!(nfmu.solution.events, e)
        end
    end

    fmi2EnterContinuousTimeMode(c)
end

# Does one step in the simulation.
function stepCompleted(nfmu::ME_NeuralFMU, x, t, integrator, tStart, tStop)

    # there might be no component!
    # @assert nfmu.currentComponent.state == fmi2ComponentStateContinuousTimeMode "stepCompleted(...): Must be in continuous time mode."

    if nfmu.progressMeter !== nothing 
        ProgressMeter.update!(nfmu.progressMeter, floor(Integer, 1000.0*(t-tStart)/(tStop-tStart)) )
    end

    if nfmu.currentComponent != nothing
        (status, enterEventMode, terminateSimulation) = fmi2CompletedIntegratorStep(nfmu.currentComponent, fmi2True)

        if enterEventMode == fmi2True
            affectFMU!(nfmu, integrator, -1)
        else
            # ToDo: set inputs here
            #fmiSetReal(myFMU, InputRef, Value)
        end
    end

    # for shadowing, we need to evaluate the jacobians in the forward pass and cache them
    # if nfmu.currentComponent.fmu.executionConfig.useComponentShadow
    #     # if no interpolation data is ready, we log the jacobians
    #     if !isPrepared(nfmu.componentShadow)
    #         evaluateJacobians(nfmu.currentComponent.fmu, x, t, setValueReferences, setValues, getValueReferences)
    #     end
    # end

end

# save FMU values 
function saveValues(nfmu, c::FMU2Component, recordValues, x, t, integrator)

    if isa(t, ForwardDiff.Dual) 
        t = ForwardDiff.value(t)
    end 
    
    if all(isa.(x, ForwardDiff.Dual))
        x = collect(ForwardDiff.value(e) for e in x)
    end

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    nfmu.neuralODE.model(x) # evaluate NeuralFMU (set new states)

    fmi2SetTime(c, t)

    (fmi2GetReal(c, recordValues)...,)
end

function fd_eltypes(e::SubArray{<:ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    return (T, V, N)
end

function fd_eltypes(e::Vector{<:ForwardDiff.Dual{T, V, N}}) where {T, V, N}
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
            #dx_tmp = collect(ForwardDiff.value(e) for e in dx)
            #fmi2GetDerivatives!(nfmu.currentComponent, dx_tmp)
            T, V, N = fd_eltypes(dx)
            dx[:] = collect(ForwardDiff.Dual{T, V, N}(V(dx_tmp[i]), ForwardDiff.partials(dx[i])    ) for i in 1:length(dx))
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

    c = nfmu.currentComponent

    if c === nothing
        # this should never happen!
        return zeros(length(x))
    end 

    ignore_derivatives() do
        
        if isa(t, ForwardDiff.Dual) 
            t = ForwardDiff.value(t)
        end 

        # if t == nfmu.tspan[1]
        #     @debug ["fx (start)"]
        # end

        # if t == nfmu.tspan[end]
        #     @debug ["fx (stop)"]
        # end

        fmi2SetTime(c, t)

    end 

    dx = nfmu.neuralODE.re(p)(x)

    # build up ẋ interpolation polynominal
    # ignore_derivatives() do
    #     if nfmu.fmu.executionConfig.useComponentShadow
    #         if nfmu.fmu.ẋy_interp == nothing
    #             if t <= nfmu.tspan[end]
    #                 if length(nfmu.fmu.t_cache) == 0 || nfmu.fmu.t_cache[end] < t
    #                     push!(nfmu.fmu.ẋy_cache, collect(ForwardDiff.value(e) for e in dx) )
    #                     push!(nfmu.fmu.t_cache, t)
    #                 end
    #             end

    #             if t >= nfmu.tspan[end] # endpoint
                    
    #             end
    #         end
    #     end
    # end 

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
        nfmu.currentComponent = nothing
    else 
        nfmu = CS_NeuralFMU{Vector{FMU2}, Vector{FMU2Component} }()
        nfmu.currentComponent = Vector{Union{FMU2Component, Nothing}}(nothing, length(fmu))
    end

    nfmu.fmu = fmu

    nfmu.model = model # Chain(model.layers...)

    nfmu.tspan = tspan
    nfmu.saveat = saveat

    nfmu
end

function prepareFMU(fmu::FMU2, c::Union{Nothing, FMU2Component}, instantiate::Union{Nothing, Bool}, terminate::Union{Nothing, Bool}, reset::Union{Nothing, Bool}, setup::Union{Nothing, Bool}, parameters::Union{Dict{<:Any, <:Any}, Nothing}, t_start, t_stop, tolerance;
    x0::Union{Array{<:Real}, Nothing}=nothing, pushComponents::Bool=true)

    c = nothing

    ignore_derivatives() do
        if instantiate === nothing 
            instantiate = fmu.executionConfig.instantiate
        end

        if terminate === nothing 
            terminate = fmu.executionConfig.terminate
        end

        if reset === nothing 
            reset = fmu.executionConfig.reset 
        end

        if setup === nothing 
            setup = fmu.executionConfig.setup 
        end 

        # instantiate (hard)
        if instantiate

            # remove old one if we missed it (callback)
            if c != nothing
                fmi2FreeInstance!(c)
                @debug "[AUTO-RELEASE INST]"
            end

            c = fmi2Instantiate!(fmu; pushComponents=pushComponents)
            @debug "[NEW INST]"
        else
            if c === nothing
                c = fmu.components[end]
            end
        end

        # soft terminate (if necessary)
        if terminate
            retcode = fmi2Terminate(c; soft=true)
            @assert retcode == fmi2StatusOK "fmi2Simulate(...): Termination failed with return code $(retcode)."
        end

        # soft reset (if necessary)
        if reset
            retcode = fmi2Reset(c; soft=true)
            @assert retcode == fmi2StatusOK "fmi2Simulate(...): Reset failed with return code $(retcode)."
        end 

        # enter setup (hard)
        if setup
            retcode = fmi2SetupExperiment(c, t_start, t_stop; tolerance=tolerance)
            @assert retcode == fmi2StatusOK "fmi2Simulate(...): Setting up experiment failed with return code $(retcode)."

            retcode = fmi2EnterInitializationMode(c)
            @assert retcode == fmi2StatusOK "fmi2Simulate(...): Entering initialization mode failed with return code $(retcode)."
        end 

        if x0 === nothing
            x0 = fmi2GetContinuousStates(c)
        else
            retcode = fmi2SetContinuousStates(c, x0)
            @assert retcode == fmi2StatusOK "fmi2Simulate(...): Setting initial state failed with return code $(retcode)."
        end

        if parameters !== nothing
            retcodes = fmi2Set(c, collect(keys(parameters)), collect(values(parameters)) )
            @assert all(retcodes .== fmi2StatusOK) "fmi2Simulate(...): Setting initial parameters failed with return code $(retcode)."
        end

        # exit setup (hard)
        if setup
            retcode = fmi2ExitInitializationMode(c)
            @assert retcode == fmi2StatusOK "fmi2Simulate(...): Exiting initialization mode failed with return code $(retcode)."
        end
    end # ignore_derivatives

    return c, x0
end

function prepareFMU(fmu::Vector{FMU2}, c::Vector{Union{Nothing, FMU2Component}}, instantiate::Union{Nothing, Bool}, terminate::Union{Nothing, Bool}, reset::Union{Nothing, Bool}, setup::Union{Nothing, Bool}, parameters::Union{Vector{Union{Dict{<:Any, <:Any}, Nothing}}, Nothing}, t_start, t_stop, tolerance;
    x0::Union{Vector{Union{Array{<:Real}, Nothing}}, Nothing}=nothing)

    ignore_derivatives() do
        for i in 1:length(fmu)

            if instantiate === nothing 
                instantiate = fmu[i].executionConfig.instantiate
            end

            if terminate === nothing 
                terminate = fmu[i].executionConfig.terminate
            end

            if reset === nothing 
                reset = fmu[i].executionConfig.reset 
            end

            if setup === nothing 
                setup = fmu[i].executionConfig.setup 
            end 

            # instantiate (hard)
            if instantiate
                # remove old one if we missed it (callback)
                if c[i] != nothing
                    fmi2FreeInstance!(c[i])
                    @debug "[AUTO-RELEASE INST]"
                end

                c[i] = fmi2Instantiate!(fmu[i])
                @debug "[NEW INST]"
            else
                if c[i] === nothing
                    c[i] = fmu[i].components[end]
                end
            end

            # soft terminate (if necessary)
            if terminate
                retcode = fmi2Terminate(c[i]; soft=true)
                @assert retcode == fmi2StatusOK "fmi2Simulate(...): Termination failed with return code $(retcode)."
            end

            # soft reset (if necessary)
            if reset
                retcode = fmi2Reset(c[i]; soft=true)
                @assert retcode == fmi2StatusOK "fmi2Simulate(...): Reset failed with return code $(retcode)."
            end 

            # enter setup (hard)
            if setup
                retcode = fmi2SetupExperiment(c[i], t_start, t_stop; tolerance=tolerance)
                @assert retcode == fmi2StatusOK "fmi2Simulate(...): Setting up experiment failed with return code $(retcode)."

                retcode = fmi2EnterInitializationMode(c[i])
                @assert retcode == fmi2StatusOK "fmi2Simulate(...): Entering initialization mode failed with return code $(retcode)."
            end 

            if x0 !== nothing 
                if x0[i] === nothing
                    x0[i] = fmi2GetContinuousStates(c[i])
                else
                    retcode = fmi2SetContinuousStates(c[i], x0[i])
                    @assert retcode == fmi2StatusOK "fmi2Simulate(...): Setting initial state failed with return code $(retcode)."
                end
            end

            if parameters !== nothing 
                if parameters[i] !== nothing
                    retcodes = fmi2Set(c[i], collect(keys(parameters[i])), collect(values(parameters[i])) )
                    @assert all(retcodes .== fmi2StatusOK) "fmi2Simulate(...): Setting initial parameters failed with return code $(retcode)."
                end
            end

            # exit setup (hard)
            if setup
                retcode = fmi2ExitInitializationMode(c[i])
                @assert retcode == fmi2StatusOK "fmi2Simulate(...): Exiting initialization mode failed with return code $(retcode)."
            end
        end

    end # ignore_derivatives

    return c, x0
end

function finishFMU(fmu::FMU2, c::Union{FMU2Component, Nothing}, freeInstance::Union{Nothing, Bool}; popComponent::Bool=true)

    ignore_derivatives() do
        if freeInstance === nothing 
            freeInstance = fmu.executionConfig.freeInstance
        end

        if c != nothing
            if freeInstance
                fmi2FreeInstance!(c; popComponent=popComponent)
                @debug "[RELEASED INST]"
            end
            c = nothing
        end

    end # ignore_derivatives

    return c
end

function finishFMU(fmu::Vector{FMU2}, c::Vector{Union{FMU2Component, Nothing}}, freeInstance::Union{Nothing, Bool})

    ignore_derivatives() do
        for i in 1:length(fmu)
            if freeInstance === nothing 
                freeInstance = fmu[i].executionConfig.freeInstance
            end
        
            if c[i] != nothing
                if freeInstance
                    fmi2FreeInstance!(c[i])
                    @debug "[RELEASED INST]"
                end
                c[i] = nothing
            end
        end

    end # ignore_derivatives

    return c
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
                              tolerance::Union{Real, Nothing} = nothing,
                              parameters::Union{Dict{<:Any, <:Any}, Nothing} = nothing,
                              setup::Union{Bool, Nothing} = nothing,
                              reset::Union{Bool, Nothing} = nothing,
                              instantiate::Union{Bool, Nothing} = nothing,
                              freeInstance::Union{Bool, Nothing} = nothing,
                              terminate::Union{Bool, Nothing} = nothing,
                              kwargs...)
    ignore_derivatives() do
        @debug "ME_NeuralFMU..."
    end

    saving = (length(nfmu.recordValues) > 0)
    
    nfmu.firstRun = true

    nfmu.solution = FMU2Solution(nfmu.fmu)

    nfmu.tolerance = tolerance
    nfmu.parameters = parameters
    nfmu.setup = setup
    nfmu.reset = reset
    nfmu.instantiate = instantiate
    nfmu.freeInstance = freeInstance
    nfmu.terminate = terminate

    nfmu.x0 = x_start

    ignore_derivatives() do
        # remove last run's shadow
        if nfmu.fmu.executionConfig.useComponentShadow
            if length(nfmu.fmu.components) > 0 && nfmu.fmu.components[end] == nfmu.currentComponent
                ind = findall(x -> x==nfmu.currentComponent, nfmu.fmu.components)

                if length(ind) == 1
                    @debug ["SHADOW CLEANUP"]
                    deleteat!(nfmu.fmu.components, ind)
                end
            end

            if nfmu.progressMeter != nothing
                ProgressMeter.finish!(nfmu.progressMeter)
            end
        end
    end

    nfmu.currentComponent = nothing
    nfmu.solveCycle = 0
    nfmu.progressMeter = nothing

    ########

    nfmu.callbacks = []
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

    ignore_derivatives() do

        # from here on, we are in event mode, if `setup=false` this is the job of the user
        # @assert nfmu.currentComponent.state == fmi2ComponentStateEventMode "FMU needs to be in event mode after setup (end)."

        # this is not necessary for NeuralFMUs:
        #x0 = fmi2GetContinuousStates(nfmu.currentComponent)
        #x0_nom = fmi2GetNominalsOfContinuousStates(nfmu.currentComponent)

        # initial event handling
        # handleEvents(nfmu.currentComponent) 
        # fmi2EnterContinuousTimeMode(nfmu.currentComponent)

        nfmu.fmu.hasStateEvents = (nfmu.fmu.modelDescription.numberOfEventIndicators > 0)
        # nfmu.fmu.hasTimeEvents = (nfmu.currentComponent.eventInfo.nextEventTimeDefined == fmi2True)

        startcb = FunctionCallingCallback((u, t, integrator) -> startCallback(nfmu, t);
                 funcat=[nfmu.tspan[1]], func_start=true, func_everystep=false)
        push!(nfmu.callbacks, startcb)

        # integrator step callback

        if showProgress

            nfmu.progressMeter = ProgressMeter.Progress(1000; desc="Simulating ME-NeuralFMU ...", color=:blue, dt=1.0) #, barglyphs=ProgressMeter.BarGlyphs("[=> ]"))
            ProgressMeter.update!(nfmu.progressMeter, 0) # show it!

            stepCb = FunctionCallingCallback((x, t, integrator) -> stepCompleted(nfmu, x, t, integrator, t_start, t_stop);
                                                func_everystep=true,
                                                func_start=true)
            push!(nfmu.callbacks, stepCb)
        end

        # state event callback
           
        if nfmu.fmu.hasStateEvents && nfmu.fmu.executionConfig.handleStateEvents
            if nfmu.fmu.executionConfig.useVectorCallbacks

                eventCb = VectorContinuousCallback((out, x, t, integrator) -> condition(nfmu, out, x, t, integrator),
                                                (integrator, idx) -> affectFMU!(nfmu, integrator, idx),
                                                Int64(nfmu.fmu.modelDescription.numberOfEventIndicators);
                                                rootfind=RightRootFind,
                                                save_positions=(false, false),
                                                interp_points=nfmu.fmu.executionConfig.rootSearchInterpolationPoints) 
                push!(nfmu.callbacks, eventCb)
            else

                for idx in 1:nfmu.fmu.modelDescription.numberOfEventIndicators
                    eventCb = ContinuousCallback((x, t, integrator) -> conditionSingle(nfmu, idx, x, t, integrator),
                                                    (integrator) -> affectFMU!(nfmu, integrator, idx);
                                                    rootfind=RightRootFind,
                                                    save_positions=(false, false),
                                                    interp_points=nfmu.fmu.executionConfig.rootSearchInterpolationPoints) 
                    push!(nfmu.callbacks, eventCb)
                end
            end
        end

        # time event handling 

        if nfmu.fmu.executionConfig.handleTimeEvents # && nfmu.fmu.hasTimeEvents
            timeEventCb = IterativeCallback((integrator) -> time_choice(nfmu, integrator),
                                            (integrator) -> affectFMU!(nfmu, integrator, 0), 
                                            Float64; 
                                            initial_affect=false,
                                            save_positions=(false,false))
        
            push!(nfmu.callbacks, timeEventCb)
        end
        
        # auto pick sensealg 

        if sense === nothing

            # currently, Callbacks only working with ForwardDiff
            if length(nfmu.callbacks) > 0 
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
                if fds_chunk_size < 1
                    fds_chunk_size = 1
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
            nfmu.solution.values = SavedValues(Float64, Tuple{collect(Float64 for i in 1:length(nfmu.recordValues))...})

            savingCB = SavingCallback((x, t, integrator) -> saveValues(nfmu, nfmu.fmu.components[end], nfmu.recordValues, x, t, integrator), 
                            nfmu.solution.values, 
                              saveat=nfmu.saveat)
            push!(nfmu.callbacks, savingCB)
        end

        # custom callbacks
        for cb in nfmu.customCallbacks
            push!(nfmu.callbacks, cb)
        end

        stopcb = FunctionCallingCallback((u, t, integrator) -> stopCallback(nfmu, t);
                                    funcat=[nfmu.tspan[end]])
        push!(nfmu.callbacks, stopcb)

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

    if (length(nfmu.callbacks) == 2) # only start and stop callback, so the system is pure continuous
        startCallback(nfmu, nfmu.tspan[1])
        nfmu.solution.states = solve(prob, nfmu.neuralODE.args...; sensealg=sense, saveat=nfmu.saveat, nfmu.neuralODE.kwargs...) 
        stopCallback(nfmu, nfmu.tspan[end])
    else
        nfmu.solution.states = solve(prob, nfmu.neuralODE.args...; sensealg=sense, saveat=nfmu.saveat, callback = CallbackSet(nfmu.callbacks...), nfmu.neuralODE.kwargs...) 
    end

    nfmu.solution.success = (nfmu.solution.states.retcode == :Success)

    # cleanup progress meter

    ignore_derivatives() do

        # this code is executed after the initial solve-evaluation (further ForwardDiff-solve-evaluations will happen after this!)

    end # ignore_derivatives

    return nfmu.solution
end

"""
Evaluates the CS_NeuralFMU in the timespan given during construction or in a custum timespan from `t_start` to `t_stop` with a given time step size `t_step`.

Via optional argument `reset`, the FMU is reset every time evaluation is started (default=`true`).
"""
function (nfmu::CS_NeuralFMU{F, C})(inputFct,
                                 t_step::Real, 
                                 t_start::Real = nfmu.tspan[1], 
                                 t_stop::Real = nfmu.tspan[end]; 
                                 tolerance::Union{Real, Nothing} = nothing,
                                 parameters::Union{Dict{<:Any, <:Any}, Nothing} = nothing,
                                 setup::Union{Bool, Nothing} = nothing,
                                 reset::Union{Bool, Nothing} = nothing,
                                 instantiate::Union{Bool, Nothing} = nothing,
                                 freeInstance::Union{Bool, Nothing} = nothing,
                                 terminate::Union{Bool, Nothing} = nothing) where {F, C}

    nfmu.solution = FMU2Solution(nfmu.fmu)

    nfmu.currentComponent, _ = prepareFMU(nfmu.fmu, nfmu.currentComponent, instantiate, terminate, reset, setup, parameters, t_start, t_stop, tolerance)

    ts = collect(t_start:t_step:t_stop)
    nfmu.currentComponent.skipNextDoStep = true # skip first fim2DoStep-call
    model_input = inputFct.(ts)
    valueStack = nfmu.model.(model_input)

    nfmu.solution.success = true
    nfmu.solution.values = SavedValues{Float64, Vector{Float64}}(ts, valueStack ) 

    # this is not possible in CS, clean-up happens at the next call
    #nfmu.currentComponent = finishFMU(nfmu.fmu, nfmu.currentComponent, freeInstance)
    
    return nfmu.solution
end
function (nfmu::CS_NeuralFMU{Vector{F}, Vector{C}})(inputFct,
                                         t_step::Real, 
                                         t_start::Real = nfmu.tspan[1], 
                                         t_stop::Real = nfmu.tspan[end]; 
                                         tolerance::Union{Real, Nothing} = nothing,
                                         parameters::Union{Vector{Union{Dict{<:Any, <:Any}, Nothing}}, Nothing} = nothing,
                                         setup::Union{Bool, Nothing} = nothing,
                                         reset::Union{Bool, Nothing} = nothing,
                                         instantiate::Union{Bool, Nothing} = nothing,
                                         freeInstance::Union{Bool, Nothing} = nothing,
                                         terminate::Union{Bool, Nothing} = nothing) where {F, C}

    nfmu.solution = FMU2Solution(nfmu.fmu)
    
    nfmu.currentComponent, _ = prepareFMU(nfmu.fmu, nfmu.currentComponent, instantiate, terminate, reset, setup, parameters, t_start, t_stop, tolerance)

    ts = collect(t_start:t_step:t_stop)
    model_input = inputFct.(ts)
    valueStack = nfmu.model.(model_input)

    nfmu.solution.success = true
    nfmu.solution.values = SavedValues{Float64, Vector{Float64}}(ts, valueStack ) 

    # this is not possible in CS, clean-up happens at the next call
    #nfmu.currentComponent = finishFMU(nfmu.fmu, nfmu.currentComponent, freeInstance)

    return nfmu.solution
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

