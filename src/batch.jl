#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMIImport.FMICore: FMUSnapshot
import FMIImport: fmi2Real, fmi2FMUstate, fmi2EventInfo, fmi2ComponentState
using DifferentialEquations.DiffEqCallbacks: FunctionCallingCallback

struct FMULoss{T}
    loss::T
    step::Integer 
    time::Real 

    function FMULoss{T}(loss::T, step::Integer=0, time::Real=time()) where {T}
        inst = new{T}(loss, step, time)
        return inst
    end

    function FMULoss(loss, step::Integer=0, time::Real=time())
        loss = unsense(loss)
        T = typeof(loss)
        inst = new{T}(loss, step, time)
        return inst
    end
end

function nominalLoss(l::FMULoss{T}) where T <: AbstractArray
    return unsense(sum(l.loss))
end

function nominalLoss(l::FMULoss{T}) where T <: Real
    return unsense(l.loss)
end

abstract type FMU2BatchElement end

mutable struct FMU2SolutionBatchElement{D} <: FMU2BatchElement

    snapshot::Union{FMUSnapshot, Nothing}

    xStart::Union{Vector{fmi2Real}, Nothing}
    xdStart::Union{Vector{D}, Nothing}

    tStart::fmi2Real 
    tStop::fmi2Real 

    # initialState::Union{fmi2FMUstate, Nothing}
    # initialComponentState::fmi2ComponentState
    # initialEventInfo::Union{fmi2EventInfo, Nothing}
    
    losses::Array{<:FMULoss} 
    step::Integer

    saveat::Union{AbstractVector{<:Real}, Nothing}
    targets::Union{AbstractArray, Nothing}
    
    indicesModel

    solution::FMU2Solution

    scalarLoss::Bool
    # canGetSetState::Bool

    function FMU2SolutionBatchElement{D}(;scalarLoss::Bool=true) where {D}
        inst = new()

        inst.snapshot = nothing
        inst.xStart = nothing
        inst.xdStart = nothing
        inst.tStart = -Inf
        inst.tStop = Inf

        # inst.initialState = nothing
        # inst.initialEventInfo = nothing 
        inst.losses = Array{FMULoss,1}()
        inst.step = 0

        inst.saveat = nothing
        inst.targets = nothing

        inst.indicesModel = nothing
        inst.scalarLoss = scalarLoss
        # inst.canGetSetState = canGetSetState

        return inst
    end
end

mutable struct FMU2EvaluationBatchElement <: FMU2BatchElement
    
    tStart::fmi2Real 
    tStop::fmi2Real 

    losses::Array{<:FMULoss} 
    step::Integer

    saveat::Union{AbstractVector{<:Real}, Nothing}
    targets::Union{AbstractArray, Nothing}
    features::Union{AbstractArray, Nothing}

    indicesModel

    result

    scalarLoss::Bool

    function FMU2EvaluationBatchElement(;scalarLoss::Bool=true)
        inst = new()
       
        inst.tStart = -Inf
        inst.tStop = Inf

        inst.losses = Array{FMULoss,1}()
        inst.step = 0

        inst.saveat = nothing
        inst.features = nothing
        inst.targets = nothing

        inst.indicesModel = nothing
        inst.result = nothing
        inst.scalarLoss = scalarLoss

        return inst
    end
end

function pasteFMUState!(fmu::FMU2, batchElement::FMU2SolutionBatchElement)
    c = getCurrentComponent(fmu)
    FMICore.apply!(c, batchElement.snapshot)
    @info "Pasting snapshot @$(batchElement.snapshot.t)"
    return nothing
end

function copyFMUState!(fmu::FMU2, batchElement::FMU2SolutionBatchElement)
    c = getCurrentComponent(fmu)
    if isnothing(batchElement.snapshot)
        batchElement.snapshot = FMICore.snapshot!(c)
        #batchElement.snapshot.t = batchElement.tStart
        @info "New snapshot @$(batchElement.snapshot.t)"
    else
        #tBefore = batchElement.snapshot.t
        FMICore.update!(c, batchElement.snapshot)
        #batchElement.snapshot.t = batchElement.tStart
        #tAfter = batchElement.snapshot.t

        # [Note] for discontinuous batches (time offsets inside batch),
        #        it might be necessary to correct the new snapshot time to fit the old one.
        # if tBefore != tAfter
        #     batchElement.snapshot.t = max(tBefore, tAfter)
        #     logInfo(fmu, "Corrected snapshot time from $(tAfter) to $(tBefore)")
        # end

        @info "Updated snapshot @$(batchElement.snapshot.t)"
    end
    return nothing
end

function run!(neuralFMU::ME_NeuralFMU, batchElement::FMU2SolutionBatchElement; nextBatchElement=nothing, kwargs...)

    neuralFMU.customCallbacksAfter = []
    neuralFMU.customCallbacksBefore = []
    
    # STOP CALLBACK
    if !isnothing(nextBatchElement) 
        stopcb = FunctionCallingCallback((u, t, integrator) -> copyFMUState!(neuralFMU.fmu, nextBatchElement);
                                    funcat=[batchElement.tStop])
        push!(neuralFMU.customCallbacksAfter, stopcb)
    end

    # on first run of the element, there is no snapshot
    writeSnapshot = nothing
    if isnothing(batchElement.snapshot) 
        c = getCurrentComponent(neuralFMU.fmu)
        writeSnapshot = FMICore.snapshot!(c)
    end

    @info "Running $(batchElement.tStart) with snapshot: $(!isnothing(batchElement.snapshot))..."
   
    batchElement.solution = neuralFMU(batchElement.xStart, (batchElement.tStart, batchElement.tStop); 
        readSnapshot=batchElement.snapshot, 
        writeSnapshot=writeSnapshot,
        saveat=batchElement.saveat, kwargs...)

    @assert batchElement.solution.states.t == batchElement.saveat "Batch element simulation failed, missmatch between `states.t` and `saveat`."

    neuralFMU.customCallbacksBefore = []
    neuralFMU.customCallbacksAfter = []

    batchElement.step += 1
    
    return batchElement.solution
end

function run!(model, batchElement::FMU2EvaluationBatchElement, p=nothing)
    if isnothing(p) # implicite parameter model
        batchElement.result = collect(model(f)[batchElement.indicesModel] for f in batchElement.features)
    else # explicite parameter model
        batchElement.result = collect(model(p)(f)[batchElement.indicesModel] for f in batchElement.features)
    end
end

function plot(batchElement::FMU2SolutionBatchElement; targets::Bool=true, plotkwargs...)

    fig = Plots.plot(; xlabel="t [s]", plotkwargs...) # , title="loss[$(batchElement.step)] = $(nominalLoss(batchElement.losses[end]))")
    for i in 1:length(batchElement.indicesModel)
        if !isnothing(batchElement.solution)
            @assert batchElement.solution.states.t == batchElement.saveat "Batch element plotting failed, missmatch between `states.t` and `saveat`."

            Plots.plot!(fig, batchElement.solution.states.t, collect(unsense(u[batchElement.indicesModel[i]]) for u in batchElement.solution.states.u), label="Simulation #$(i)")
        end
        if targets
            Plots.plot!(fig, batchElement.saveat, collect(d[i] for d in batchElement.targets), label="Targets #$(i)")
        end
    end

    return fig
end

function plot(batchElement::FMU2BatchElement; targets::Bool=true, features::Bool=true, plotkwargs...)

    fig = Plots.plot(; xlabel="t [s]", plotkwargs...) # , title="loss[$(batchElement.step)] = $(nominalLoss(batchElement.losses[end]))")

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
    ys = collect((length(b.losses) > 0 ? nominalLoss(b.losses[end]) : 0.0) for b in batch)

    fig = Plots.plot(; xlabel="Batch ID", ylabel="Loss", plotkwargs...)

    if plot_shadow 
        ys_shadow = collect((length(b.losses) > 1 ? nominalLoss(b.losses[end-1]) : 0.0) for b in batch)

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

function loss!(batchElement::FMU2SolutionBatchElement, lossFct; logLoss::Bool=true)

    loss = nothing

    if hasmethod(lossFct, Tuple{FMU2Solution})
        loss = lossFct(batchElement.solution)

    elseif hasmethod(lossFct, Tuple{FMU2Solution, Union{}})
        loss = lossFct(batchElement.solution, batchElement.targets)

    else # hasmethod(lossFct, Tuple{Union{}, Union{}})

        if batchElement.solution.success
            if batchElement.scalarLoss
                for i in 1:length(batchElement.indicesModel)
                    dataTarget = collect(d[i] for d in batchElement.targets)
                    modelOutput = collect(u[batchElement.indicesModel[i]] for u in batchElement.solution.states.u)

                    if isnothing(loss)
                        loss = lossFct(modelOutput, dataTarget)
                    else
                        loss += lossFct(modelOutput, dataTarget)
                    end
                end
            else
                dataTarget = batchElement.targets
                modelOutput = collect(u[batchElement.indicesModel] for u in batchElement.solution.states.u)

                if isnothing(loss)
                    loss = lossFct(modelOutput, dataTarget)
                else
                    loss += lossFct(modelOutput, dataTarget)
                end
            end
        else
            @warn "Can't compute loss for batch element, because solution is invalid (`success=false`) for batch element\n$(batchElement)."
        end
       
    end

    ignore_derivatives() do 
        if logLoss
            push!(batchElement.losses, FMULoss(loss, batchElement.step))
        end
    end

    return loss
end

function loss!(batchElement::FMU2EvaluationBatchElement, lossFct; logLoss::Bool=true)

    loss = nothing

    if batchElement.scalarLoss
        for i in 1:length(batchElement.indicesModel)
            dataTarget = collect(d[i] for d in batchElement.targets)
            modelOutput = collect(u[i] for u in batchElement.result)

            if isnothing(loss)
                loss = lossFct(modelOutput, dataTarget)
            else
                loss += lossFct(modelOutput, dataTarget)
            end
        end
    else
        dataTarget = batchElement.targets
        modelOutput = batchElement.result

        if isnothing(loss)
            loss = lossFct(modelOutput, dataTarget)
        else
            loss += lossFct(modelOutput, dataTarget)
        end
    end
     
    ignore_derivatives() do 
        if logLoss
            push!(batchElement.losses, FMULoss(loss, batchElement.step))
        end
    end

    return loss
end

function _batchDataSolution!(batch::AbstractArray{<:FMIFlux.FMU2SolutionBatchElement}, neuralFMU::NeuralFMU, x0_fun, train_t::AbstractArray{<:AbstractArray{<:Real}}, targets::AbstractArray; kwargs...)

    len = length(train_t)
    for i in 1:len 
        _batchDataSolution!(batch, neuralFMU, x0_fun, train_t[i], targets[i]; kwargs...)
    end
    return nothing
end

function _batchDataSolution!(batch::AbstractArray{<:FMIFlux.FMU2SolutionBatchElement}, neuralFMU::NeuralFMU, x0_fun, train_t::AbstractArray{<:Real}, targets::AbstractArray; 
    batchDuration::Real=(train_t[end]-train_t[1]), indicesModel=1:length(targets[1]), plot::Bool=false, scalarLoss::Bool=true)

    @assert length(train_t) == length(targets) "Timepoints in `train_t` ($(length(train_t))) must match number of `targets` ($(length(targets)))"

    canGetSetState = fmi2CanGetSetState(neuralFMU.fmu)
    if !canGetSetState
        logWarning(neuralFMU.fmu, "This FMU can't set/get a FMU state. This is suboptimal for batched training.")
    end

    # c, _ = prepareSolveFMU(neuralFMU.fmu, nothing, neuralFMU.fmu.type, nothing, nothing, nothing, nothing, nothing, nothing, neuralFMU.tspan[1], neuralFMU.tspan[end], nothing; handleEvents=FMIFlux.handleEvents)
    
    # indicesData = 1:1

    tStart = train_t[1]
    
    # iStart = timeToIndex(train_t, tStart)
    # iStop = timeToIndex(train_t, tStart + batchDuration)
    
    # startElement = FMIFlux.FMU2SolutionBatchElement(;scalarLoss=scalarLoss)
    # startElement.tStart = train_t[iStart]
    # startElement.tStop = train_t[iStop]
    # startElement.xStart = x0_fun(tStart)
    
    # startElement.saveat = train_t[iStart:iStop]
    # startElement.targets = targets[iStart:iStop]
    
    # startElement.indicesModel = indicesModel

    # push!(batch, startElement)
    
    numElements = floor(Integer, (train_t[end]-train_t[1])/batchDuration)

    D = eltype(neuralFMU.fmu.modelDescription.discreteStateValueReferences)

    for i in 1:numElements

        element = FMIFlux.FMU2SolutionBatchElement{D}(;scalarLoss=scalarLoss)
    
        iStart = FMIFlux.timeToIndex(train_t, tStart + (i-1) * batchDuration)
        iStop = FMIFlux.timeToIndex(train_t, tStart + i * batchDuration)

        element.tStart = train_t[iStart]
        element.tStop = train_t[iStop]
        element.xStart = x0_fun(element.tStart)
        
        element.saveat = train_t[iStart:iStop]
        element.targets = targets[iStart:iStop]
        
        element.indicesModel = indicesModel

        push!(batch, element)
    end

    return nothing
end

function batchDataSolution(neuralFMU::NeuralFMU, x0_fun, train_t, targets; 
    batchDuration::Real=(train_t[end]-train_t[1]), 
    indicesModel=1:length(targets[1]), 
    plot::Bool=false, 
    scalarLoss::Bool=true, 
    restartAtJump::Bool=true, 
    solverKwargs...)

    batch = Array{FMIFlux.FMU2SolutionBatchElement,1}()
    _batchDataSolution!(batch, neuralFMU, x0_fun, train_t, targets; batchDuration=batchDuration, indicesModel=indicesModel, plot=plot, scalarLoss=scalarLoss)

    numElements = length(batch)
    for i in 1:numElements
        
        nextBatchElement = nothing 
        if i < numElements && batch[i].tStop == batch[i+1].tStart
            nextBatchElement = batch[i+1]
        end 
       
        FMIFlux.run!(neuralFMU, batch[i]; nextBatchElement=nextBatchElement, solverKwargs...)
    
        if plot 
            fig = FMIFlux.plot(batch[i])
            display(fig)
        end
    end

    return batch
end

function batchDataEvaluation(train_t::AbstractArray{<:Real}, targets::AbstractArray, features::Union{AbstractArray, Nothing}=nothing; 
    batchDuration::Real=(train_t[end]-train_t[1]), indicesModel=1:length(targets[1]), plot::Bool=false, round_digits=3, scalarLoss::Bool=true)

    batch = Array{FMIFlux.FMU2EvaluationBatchElement,1}()
    
    indicesData = 1:1

    tStart = train_t[1]
    
    iStart = timeToIndex(train_t, tStart)
    iStop = timeToIndex(train_t, tStart + batchDuration)
    
    startElement = FMIFlux.FMU2EvaluationBatchElement(;scalarLoss=scalarLoss)
    startElement.tStart = train_t[iStart]
    startElement.tStop = train_t[iStop]
    
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
        push!(batch, FMIFlux.FMU2EvaluationBatchElement(;scalarLoss=scalarLoss))
    
        iStart = timeToIndex(train_t, tStart + (i-1) * batchDuration)
        iStop = timeToIndex(train_t, tStart + i * batchDuration)

        batch[i].tStart = train_t[iStart]
        batch[i].tStop = train_t[iStop]

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


