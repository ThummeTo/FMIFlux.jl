#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMIImport: fmi2Real, fmi2FMUstate, fmi2EventInfo
import ChainRulesCore: ignore_derivatives
using DiffEqCallbacks: FunctionCallingCallback
using FMIImport.ForwardDiff

struct FMU2Loss 
    loss::Real
    step::Integer 
    time::Real 

    function FMU2Loss(loss::Real, step::Integer=0, time::Real=time())
        inst = new(ForwardDiff.value(loss), step, time)
        return inst
    end
end

abstract type FMU2BatchElement end

mutable struct FMU2SolutionBatchElement <: FMU2BatchElement
    xStart::Union{AbstractVector{<:fmi2Real}, Nothing}
    tStart::fmi2Real 
    tStop::fmi2Real 

    initialState::Union{fmi2FMUstate, Nothing}
    initialEventInfo::Union{fmi2EventInfo, Nothing}
    losses::Array{<:FMU2Loss} 
    step::Integer

    saveat::Union{AbstractVector{<:Real}, Nothing}
    targets::Union{AbstractArray, Nothing}
    
    indicesModel

    solution::FMU2Solution

    function FMU2SolutionBatchElement()
        inst = new()
        inst.xStart = nothing
        inst.tStart = -Inf
        inst.tStop = Inf

        inst.initialState = nothing
        inst.initialEventInfo = nothing 
        inst.losses = Array{FMU2Loss,1}()
        inst.step = 0

        inst.saveat = nothing
        inst.targets = nothing

        inst.indicesModel = nothing

        return inst
    end
end

mutable struct FMU2EvaluationBatchElement <: FMU2BatchElement
    
    tStart::fmi2Real 
    tStop::fmi2Real 

    losses::Array{<:FMU2Loss} 
    step::Integer

    saveat::Union{AbstractVector{<:Real}, Nothing}
    targets::Union{AbstractArray, Nothing}
    features::Union{AbstractArray, Nothing}

    indicesModel

    result

    function FMU2EvaluationBatchElement()
        inst = new()
       
        inst.tStart = -Inf
        inst.tStop = Inf

        inst.losses = Array{FMU2Loss,1}()
        inst.step = 0

        inst.saveat = nothing
        inst.features = nothing
        inst.targets = nothing

        inst.indicesModel = nothing
        inst.result = nothing

        return inst
    end
end

function startStateCallback(fmu, batchElement)
    #print("Setting state ... ")

    c = getCurrentComponent(fmu)
    
    if batchElement.initialState != nothing
        fmi2SetFMUstate(c, batchElement.initialState)
        c.eventInfo = deepcopy(batchElement.initialEventInfo)
        c.t = batchElement.tStart
    else
        batchElement.initialState = fmi2GetFMUstate(c)
        batchElement.initialEventInfo = deepcopy(c.eventInfo)
        @warn "Batch element does not provide a `initialState`, I try to simulate anyway. InitialState is overwritten."
    end
end

function stopStateCallback(fmu, batchElement)
    #print("\nGetting state ... ")

    c = getCurrentComponent(fmu)
   
    if batchElement.initialState != nothing
        fmi2GetFMUstate!(c, Ref(batchElement.initialState))
    else
        batchElement.initialState = fmi2GetFMUstate(c)
    end
    batchElement.initialEventInfo = deepcopy(c.eventInfo)
    
    #println("done @ $(batchElement.initialState) in componentState: $(c.state)!")
end

function run!(neuralFMU::ME_NeuralFMU, batchElement::FMU2SolutionBatchElement; lastBatchElement=nothing, kwargs...)

    ignore_derivatives() do

        neuralFMU.customCallbacksAfter = []
        neuralFMU.customCallbacksBefore = []

        # START CALLBACK
        startcb = FunctionCallingCallback((u, t, integrator) -> startStateCallback(neuralFMU.fmu, batchElement);
                funcat=[batchElement.tStart], func_start=true, func_everystep=false)
        push!(neuralFMU.customCallbacksBefore, startcb)
        
        # STOP CALLBACK
        if lastBatchElement != nothing
            stopcb = FunctionCallingCallback((u, t, integrator) -> stopStateCallback(neuralFMU.fmu, lastBatchElement);
                                        funcat=[batchElement.tStop])
            push!(neuralFMU.customCallbacksAfter, stopcb)
        end
    end

    batchElement.solution = neuralFMU(batchElement.xStart, (batchElement.tStart, batchElement.tStop); saveat=batchElement.saveat, kwargs...)

    ignore_derivatives() do 

        if lastBatchElement != nothing
            #lastBatchElement.tStart = ForwardDiff.value(batchElement.solution.states.t[end])
            #lastBatchElement.xStart = collect(ForwardDiff.value(u) for u in batchElement.solution.states.u[end])
        end

        neuralFMU.customCallbacksBefore = []
        neuralFMU.customCallbacksAfter = []

        batchElement.step += 1
    end

    return batchElement.solution
end

function run!(model, batchElement::FMU2EvaluationBatchElement, p=nothing)
    if isnothing(p) # implicite parameter model
        batchElement.result = collect(model(f)[batchElement.indicesModel] for f in batchElement.features)
    else # explicite parameter model
        batchElement.result = collect(model(p)(f)[batchElement.indicesModel] for f in batchElement.features)
    end
end

function plot(batchElement::FMU2SolutionBatchElement; targets::Bool=true, kwargs...)

    fig = Plots.plot(; xlabel="t [s]") # , title="loss[$(batchElement.step)] = $(batchElement.losses[end].loss)")
    for i in 1:length(batchElement.indicesModel)
        if batchElement.solution != nothing
            Plots.plot!(fig, batchElement.saveat, collect(ForwardDiff.value(u[batchElement.indicesModel[i]]) for u in batchElement.solution.states.u), label="Simulation #$(i)")
        end
        if targets
            Plots.plot!(fig, batchElement.saveat, collect(d[i] for d in batchElement.targets), label="Targets #$(i)")
        end
    end

    return fig
end

function plot(batchElement::FMU2BatchElement; targets::Bool=true, features::Bool=true, kwargs...)

    fig = Plots.plot(; xlabel="t [s]") # , title="loss[$(batchElement.step)] = $(batchElement.losses[end].loss)")

    if batchElement.features != nothing && features
        for i in 1:length(batchElement.features[1])
            Plots.plot!(fig, batchElement.saveat, collect(d[i] for d in batchElement.features), style=:dash, label="Features #$(i)")
        end
    end

    for i in 1:length(batchElement.indicesModel)
        if batchElement.result != nothing
            Plots.plot!(fig, batchElement.saveat, collect(ForwardDiff.value(u[i]) for u in batchElement.result), label="Evaluation #$(i)")
        end
        if targets 
            Plots.plot!(fig, batchElement.saveat, collect(d[i] for d in batchElement.targets), label="Targets #$(i)")
        end
    end

    return fig
end

function plot(batch::AbstractArray{<:FMU2BatchElement}; plot_mean::Bool=true, plot_shadow::Bool=true, plotkwargs...)

    num = length(batch)

    xs = 1:num 
    ys = collect((length(b.losses) > 0 ? b.losses[end].loss : 0.0) for b in batch)

    fig = Plots.plot(; xlabel="Batch ID", ylabel="Loss", plotkwargs...)

    if plot_shadow 
        ys_shadow = collect((length(b.losses) > 1 ? b.losses[end-1].loss : 0.0) for b in batch)

        Plots.bar!(fig, xs, ys_shadow; label="Previous loss", color=:green, bar_width=1.0);
    end

    Plots.bar!(fig, xs, ys; label="Current loss", color=:blue, bar_width=0.5);
    
    if plot_mean
        avgsum = mean(ys)
        Plots.plot!(fig, [1,num], [avgsum, avgsum]; label="mean")
    end
    
    return fig
end

function plotLoss(batchElement::FMU2BatchElement; xaxis::Symbol = :steps)

    @assert length(batchElement.losses) > 0 "Can't plot, no losses!"
    
    ts = nothing 
    tlabel = "" 

    if xaxis == :time
        ts = collect(l.time for l in batchElement.losses)
        tlabel = "t [s]"
    elseif xaxis == :steps 
        ts = collect(l.step for l in batchElement.losses)
        tlabel = "steps [/]"
    else
        @assert false "unsupported keyword for `xaxis`."
    end
    ls = collect(l.loss for l in batchElement.losses)

    fig = Plots.plot(ts, ls, xlabel=tlabel, ylabel="Loss")

    return fig
end

function loss!(batchElement::FMU2BatchElement, lossFct; logLoss::Bool=true)

    loss = 0.0

    for i in 1:length(batchElement.indicesModel)
        modelOutput = nothing 
        
        if isa(batchElement, FMU2SolutionBatchElement)
            modelOutput = collect(u[batchElement.indicesModel[i]] for u in batchElement.solution.states.u)
        else
            modelOutput = collect(u[i] for u in batchElement.result)
        end
        dataTarget = collect(d[i] for d in batchElement.targets)

        loss += lossFct(modelOutput, dataTarget)
    end

    ignore_derivatives() do 
        if logLoss
            push!(batchElement.losses, FMU2Loss(loss, batchElement.step))
        end
    end

    return loss
end

function batchDataSolution(neuralFMU::NeuralFMU, x0_fun, train_t::AbstractArray{<:Real}, targets::AbstractArray; 
    batchDuration::Real=(train_t[end]-train_t[1]), indicesModel=1:length(targets[1]), plot::Bool=false, solverKwargs...)

    if fmi2CanGetSetState(neuralFMU.fmu)
        @assert !neuralFMU.fmu.executionConfig.instantiate "Batching not possible for auto-instanciating FMUs."
    else
        @warn "This FMU can't set/get a FMU state. So discrete states can't be estimated together with the continuous solution." 
    end

    batch = Array{FMIFlux.FMU2SolutionBatchElement,1}()
    
    indicesData = 1:1

    tStart = train_t[1]
    
    iStart = timeToIndex(train_t, tStart)
    iStop = timeToIndex(train_t, tStart + batchDuration)
    
    startElement = FMIFlux.FMU2SolutionBatchElement()
    startElement.tStart = tStart 
    startElement.tStop = tStart + batchDuration
    startElement.xStart = x0_fun(tStart)
    
    startElement.saveat = train_t[iStart:iStop]
    startElement.targets = targets[iStart:iStop]
    
    startElement.indicesModel = indicesModel

    push!(batch, startElement)
    
    for i in 2:floor(Integer, (train_t[end]-train_t[1])/batchDuration)
        push!(batch, FMIFlux.FMU2SolutionBatchElement())
    
        FMIFlux.run!(neuralFMU, batch[i-1]; lastBatchElement=batch[i], solverKwargs...)
    
        # overwrite start state
        batch[i].tStart = tStart + (i-1) * batchDuration
        batch[i].tStop = tStart + i * batchDuration
        batch[i].xStart = x0_fun(batch[i].tStart)
    
        iStart = timeToIndex(train_t, batch[i].tStart)
        iStop = timeToIndex(train_t, batch[i].tStop)
        batch[i].saveat = train_t[iStart:iStop]
        batch[i].targets = targets[iStart:iStop]
        
        batch[i].indicesModel = indicesModel
    
        if plot
            fig = FMIFlux.plot(batch[i-1]; solverKwargs...)
            display(fig)
        end
    end

    return batch
end

function batchDataEvaluation(train_t::AbstractArray{<:Real}, targets::AbstractArray, features::Union{AbstractArray, Nothing}=nothing; 
    batchDuration::Real=(train_t[end]-train_t[1]), indicesModel=1:length(targets[1]), plot::Bool=false)

    batch = Array{FMIFlux.FMU2EvaluationBatchElement,1}()
    
    indicesData = 1:1

    tStart = train_t[1]
    
    iStart = timeToIndex(train_t, tStart)
    iStop = timeToIndex(train_t, tStart + batchDuration)
    
    startElement = FMIFlux.FMU2EvaluationBatchElement()
    startElement.tStart = tStart 
    startElement.tStop = tStart + batchDuration
    
    startElement.saveat = train_t[iStart:iStop]
    startElement.targets = targets[iStart:iStop]
    if features != nothing 
        startElement.features = features[iStart:iStop]
    else
        startElement.features = startElement.targets
    end
    startElement.indicesModel = indicesModel

    push!(batch, startElement)
    
    for i in 2:floor(Integer, (train_t[end]-train_t[1])/batchDuration)
        push!(batch, FMIFlux.FMU2EvaluationBatchElement())
    
        # overwrite start state
        batch[i].tStart = tStart + (i-1) * batchDuration
        batch[i].tStop = tStart + i * batchDuration
        
        iStart = timeToIndex(train_t, batch[i].tStart)
        iStop = timeToIndex(train_t, batch[i].tStop)
        batch[i].saveat = train_t[iStart:iStop]
        batch[i].targets = targets[iStart:iStop]
        if features != nothing 
            batch[i].features = features[iStart:iStop]
        else
            batch[i].features = batch[i].targets
        end
        batch[i].indicesModel = indicesModel
    
        if plot
            fig = FMIFlux.plot(batch[i-1])
            display(fig)
        end
    end

    return batch
end


