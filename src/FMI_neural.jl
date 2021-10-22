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

# time caching (to set correct time in ME-NeuralFMUs)
function NeuralFMUGetCachedTime(fmu::FMU)
    Zygote.ignore() do
        return fmu.next_t
    end

    return 0.0
end

function NeuralFMUCacheTime(fmu::FMU, 
                            t::Real)
    fmu.next_t = t
    nothing
end
function NeuralFMUCacheTime_Gradient(c̄, 
                                     fmu::FMU, 
                                     t::Real)
    tuple(0.0, 0.0)
end
@adjoint NeuralFMUCacheTime(fmu, t) = NeuralFMUCacheTime(fmu, t), c̄ -> NeuralFMUCacheTime_Gradient(c̄, fmu, t)


# state caching (to set correct state in ME-NeuralFMUs)
function NeuralFMUCacheState(fmu::FMU, 
                             x::Array{<:Real})
    fmu.next_x = x
end
function NeuralFMUCacheState_Gradient(c̄, 
                                      fmu::FMU, 
                                      x::Array{<:Real})
    tuple(0.0, 0.0)
end
@adjoint NeuralFMUCacheState(fmu, x) = NeuralFMUCacheState(fmu, x), c̄ -> NeuralFMUCacheState_Gradient(c̄, fmu, x)

# state derivative caching (to set correct state derivative in ME-NeuralFMUs)
function NeuralFMUCacheStateDerivative(fmu::FMU, 
                                       dx::Array{<:Real})
    fmu.next_dx = dx
end
function NeuralFMUCacheStateDerivative_Gradient(c̄, 
                                                fmu::FMU, 
                                                dx::Array{<:Real})
    tuple(0.0, 0.0)
end
@adjoint NeuralFMUCacheStateDerivative(fmu, dx) = NeuralFMUCacheStateDerivative(fmu, dx), c̄ -> NeuralFMUCacheStateDerivative_Gradient(c̄, fmu, dx)

# helper to add an additional time state later used to setup time inside the FMU
function NeuralFMUInputLayer(fmu::FMU, 
                             inputs::Array{<:Real})
    t = inputs[1]
    x = inputs[2:end]
    NeuralFMUCacheTime(fmu, t)
    x
end

function NeuralFMUInputLayer(fmus::Vector{T}, inputs) where {T}
    t = inputs[1]
    x = inputs[2:end]
    for fmu in fmus
        NeuralFMUCacheTime(fmu, t)
    end
    return x
end

# helper to add an additional time state derivative (for ME-NeuralFMUs)
function NeuralFMUOutputLayerME(inputs::Array{<:Real})
    dt = 1.0
    dx = inputs
    vcat([dt], dx)
end

# helper to add an additional time state derivative (for CS-NeuralFMUs)
function NeuralFMUOutputLayerCS(fmu::FMU, 
                                inputs::Array{<:Real})
    out = inputs
    t = fmu.t 
    vcat([t], out)
end

function NeuralFMUOutputLayerCS(fmus::Vector{T}, inputs) where {T}
    out = inputs
    t = fmus[1].t
    for fmu in fmus
        @assert t == fmu.t
    end
    vcat([t], out)
end

# function that saves periodically FMU values (in case you are interested in any)
function saveFunc(nfmu, u, t, integrator)
    values = fmi2GetReal(nfmu.fmu, nfmu.recordValues)
    return (values...,)
end

"""
Constructs a ME-NeuralFMU where the FMU is at an unknown location inside of the NN.

Arguents:
    - `fmu` the considered FMU inside the NN 
    - `model` the NN topology (e.g. Flux.chain)
    - `tspan` simulation time span
    - `alg` a numerical ODE solver

Keyword arguments:
    - `saveat` time points to save the NeuralFMU output, if empty, solver step size is used (may be non-equidistant)
    - `addTopLayer` adds an input layer to add a time state (default=`true`) to set the FMU-time
    - `addBottomLayer` adds an input layer to remove the time state (default=`true`)
    - `fixstep` forces fixed step integration
    - `recordFMUValues` additionally records internal FMU variables (currently not supported because of open issues)
"""
function ME_NeuralFMU(fmu, 
                      model, 
                      tspan, 
                      alg=nothing; 
                      saveat=[], 
                      addTopLayer = true, 
                      addBottomLayer = true, 
                      recordFMUValues = [], 
                      fixstep=0.0)

    nfmu = ME_NeuralFMU()
    nfmu.fmu = fmu

    ext_model = nothing

    if addTopLayer && addBottomLayer
        ext_model = Chain(inputs -> NeuralFMUInputLayer(nfmu.fmu, inputs),
                          model.layers...,
                          inputs -> NeuralFMUOutputLayerME(inputs))
    elseif addTopLayer
        ext_model = Chain(inputs -> NeuralFMUInputLayer(nfmu.fmu, inputs),
                          model.layers...)
    elseif addBottomLayer
        ext_model = Chain(model.layers...,
                          inputs -> NeuralFMUOutputLayerME(inputs))
    else
        ext_model = Chain(model.layers...)
    end

    nfmu.recordValues = recordFMUValues
    if length(nfmu.recordValues) > 0
        @assert false "ME_NeuralFMU(...): keyword `recordFMUValues` is currently not supported (under development). Please remove keyword."

        tmp = zeros(Float64, length(nfmu.recordValues))
        type = typeof((tmp...,))

        nfmu.saved_values = SavedValues(Float64, type)
        cb = SavingCallback((u,t,integrator)->saveFunc(nfmu, u, t, integrator), nfmu.saved_values; saveat=saveat)

        if fixstep > 0.0
            nfmu.neuralODE = NeuralODE(ext_model, tspan, alg, callback=cb, dt=fixstep, adaptive=false)
        else
            nfmu.neuralODE = NeuralODE(ext_model, tspan, alg, callback=cb)
        end
    else
        if fixstep > 0.0
            nfmu.neuralODE = NeuralODE(ext_model, tspan, alg; saveat=saveat, dt=fixstep, adaptive=false)
        else
            nfmu.neuralODE = NeuralODE(ext_model, tspan, alg; saveat=saveat)
        end
    end

    nfmu.tspan = tspan
    nfmu.saveat = saveat

    nfmu
end

"""
Constructs a CS-NeuralFMU where the FMU is at an unknown location inside of the NN.

Arguents:
    - `fmu` the considered FMU inside the NN 
    - `model` the NN topology (e.g. Flux.chain)
    - `tspan` simulation time span

Keyword arguments:
    - `saveat` time points to save the NeuralFMU output, if empty, solver step size is used (may be non-equidistant)
    - `addTopLayer` adds an input layer to add a time state (default) to set the FMU-time
    - `addBottomLayer` adds an input layer to remove the time state (default)
"""
function CS_NeuralFMU(fmu, 
                      model, 
                      tspan; 
                      saveat=[], 
                      addTopLayer = true, 
                      addBottomLayer = true, 
                      recordValues = [])

    nfmu = CS_NeuralFMU{typeof(fmu)}()

    nfmu.fmu = fmu

    if addTopLayer && addBottomLayer
        nfmu.model = Chain(inputs -> NeuralFMUInputLayer(nfmu.fmu, inputs),
                          model.layers...,
                          inputs -> NeuralFMUOutputLayerCS(nfmu.fmu, inputs))
    elseif addTopLayer
        nfmu.model = Chain(inputs -> NeuralFMUInputLayer(nfmu.fmu, inputs),
                          model.layers...)
    elseif addBottomLayer
        nfmu.model = Chain(model.layers...,
                          inputs -> NeuralFMUOutputLayerCS(nfmu.fmu, inputs))
    else
        nfmu.model = model.layers
    end

    nfmu.tspan = tspan
    nfmu.saveat = saveat

    nfmu
end

"""
Evaluates the ME_NeuralFMU in the timespan given during construction or in a custum timespan from `t_start` to `t_stop` for a given start state `x_start`.

Via optional argument `reset`, the FMU is reset every time evaluation is started (default=`true`).
"""
function (nfmu::ME_NeuralFMU)(x_start::Array{<:Real}, 
                              t_start::Real = nfmu.tspan[1], 
                              t_stop = nfmu.tspan[end]; 
                              reset::Bool=true)

    if reset
        fmiReset(nfmu.fmu)
        fmiSetupExperiment(nfmu.fmu, t_start)
        fmiEnterInitializationMode(nfmu.fmu)
        fmiExitInitializationMode(nfmu.fmu)
    end

    x0 = [t_start, x_start...]

    nfmu.solution = nfmu.neuralODE(x0)

    # DEBUGGING: Code from DiffEqFlux
    #p = nfmu.neuralODE.p
    #dudt_(u,p,t) = nfmu.neuralODE.re(p)(u)
    #ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    #prob = ODEProblem{false}(ff,x0,getfield(nfmu.neuralODE,:tspan),p)
    #sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())

    #sa = t_start:Float64(nfmu.saveat.step):t_stop
    #nfmu.solution = solve(prob,nfmu.neuralODE.args...;sense=sense,saveat=sa)
   
    nfmu.solution
end

"""
Evaluates the CS_NeuralFMU in the timespan given during construction or in a custum timespan from `t_start` to `t_stop` with a given time step size `t_step`.

Via optional argument `reset`, the FMU is reset every time evaluation is started (default=`true`).
"""
function (nfmu::CS_NeuralFMU{T})(t_step::Real, 
                              t_start::Real = nfmu.tspan[1], 
                              t_stop::Real = nfmu.tspan[end]; 
                              inputs::Array{<:Real} = zeros(Real,0), 
                              reset::Bool = true) where {T}

    if reset
        while fmiReset(nfmu.fmu) != 0
        end
        while fmiSetupExperiment(nfmu.fmu, t_start) != 0
        end
        while fmiEnterInitializationMode(nfmu.fmu) != 0
        end
        while fmiExitInitializationMode(nfmu.fmu) != 0
        end
    end

    ts = t_start:t_step:t_stop
    modelInput = collect.(eachrow(hcat(ts, inputs)))
    nfmu.valueStack = nfmu.model.(modelInput)

    nfmu.valueStack
end

function (nfmu::CS_NeuralFMU{Vector{T}})(t_step, t_start=nfmu.tspan[1], t_stop=nfmu.tspan[end]; inputs=[], reset::Bool=true) where {T}
    if reset
        for fmu in nfmu.fmu
            while fmiReset(fmu) !=0
            end
            while fmiSetupExperiment(fmu, t_start) !=0
            end
            while fmiEnterInitializationMode(fmu) !=0
            end
            while fmiExitInitializationMode(fmu) !=0
            end
        end
    end

    ts = t_start:t_step:t_stop

    model_input = collect.(eachrow(hcat(ts, inputs)))
    nfmu.valueStack = nfmu.model.(model_input)

    return nfmu.valueStack
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
                     dt::Real; 
                     setValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0), 
                     setValues::Array{<:Real} = zeros(Real, 0),
                     getValueReferences::Array{fmi2ValueReference} = zeros(fmi2ValueReference, 0))
    fmi2DoStepCS(fmu, dt;
                setValueReferences = setValueReferences,
                setValues = setValues,
                getValueReferences = getValueReferences)
end

"""
Wrapper. Call ```fmi2InputDoStepCSOutput``` for more information.
"""
function fmiInputDoStepCSOutput(fmu::FMU2, 
                                dt::Real, 
                                u::Array{<:Real})
    fmi2InputDoStepCSOutput(fmu, dt, u)
end

# define neutral gradients (=feed-trough) for ccall-functions
function neutralGradient(c̄)
    c̄
end

@adjoint fmiSetupExperiment(fmu, startTime, stopTime) = fmiSetupExperiment(fmu, startTime, stopTime), c̄ -> neutralGradient(c̄)
@adjoint fmiEnterInitializationMode(fmu) = fmiEnterInitializationMode(fmu), c̄ -> neutralGradient(c̄)
@adjoint fmiExitInitializationMode(fmu) = fmiExitInitializationMode(fmu), c̄ -> neutralGradient(c̄)
@adjoint fmiReset(fmu) = fmiReset(fmu), c̄ -> neutralGradient(c̄)
