#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMIImport.FMIBase: FMUSnapshot
import FMIImport: fmi2Real, fmi2FMUstate, fmi2EventInfo, fmi2ComponentState
using FMIImport.FMIBase.DiffEqCallbacks: FunctionCallingCallback

abstract type FMU2BatchElement end

mutable struct FMULoss{T}
    loss::T
    step::Integer
    time::Real

    function FMULoss{T}(loss::T, step::Integer = 0, time::Real = time()) where {T}
        inst = new{T}(loss, step, time)
        return inst
    end

    function FMULoss(loss, step::Integer = 0, time::Real = time())
        loss = unsense(loss)
        T = typeof(loss)
        inst = new{T}(loss, step, time)
        return inst
    end
end

function nominalLoss(l::FMULoss{T}) where {T<:AbstractArray}
    return unsense(sum(l.loss))
end

function nominalLoss(l::FMULoss{T}) where {T<:Real}
    return unsense(l.loss)
end

function nominalLoss(::Nothing)
    return Inf
end

function nominalLoss(b::FMU2BatchElement)
    return nominalLoss(b.loss)
end

mutable struct FMU2SolutionBatchElement{D} <: FMU2BatchElement

    snapshot::Union{FMUSnapshot,Nothing}

    xStart::Union{Vector{fmi2Real},Nothing}
    xdStart::Union{Vector{D},Nothing}

    tStart::fmi2Real
    tStop::fmi2Real

    # initialState::Union{fmi2FMUstate, Nothing}
    # initialComponentState::fmi2ComponentState
    # initialEventInfo::Union{fmi2EventInfo, Nothing}

    loss::FMULoss   # the current loss
    losses::Array{<:FMULoss}  # logged losses (if used)
    step::Integer

    saveat::Union{AbstractVector{<:Real},Nothing}
    targets::Union{AbstractArray,Nothing}

    indicesModel::Any

    store_result::Bool
    result::FMUSolution

    scalarLoss::Bool
    # canGetSetState::Bool

    function FMU2SolutionBatchElement{D}(; scalarLoss::Bool = true, store_result::Bool=false) where {D}
        inst = new()

        inst.snapshot = nothing
        inst.xStart = nothing
        inst.xdStart = nothing
        inst.tStart = -Inf
        inst.tStop = Inf

        # inst.initialState = nothing
        # inst.initialEventInfo = nothing 
        inst.loss = FMULoss(Inf)
        inst.losses = Array{FMULoss,1}()
        inst.step = 0

        inst.saveat = nothing
        inst.targets = nothing

        inst.indicesModel = nothing
        inst.scalarLoss = scalarLoss
        inst.store_result = store_result
        # inst.canGetSetState = canGetSetState

        return inst
    end
end

mutable struct FMU2EvaluationBatchElement <: FMU2BatchElement

    tStart::fmi2Real
    tStop::fmi2Real

    loss::FMULoss
    losses::Array{<:FMULoss}
    step::Integer

    saveat::Union{AbstractVector{<:Real},Nothing}
    targets::Union{AbstractArray,Nothing}
    features::Union{AbstractArray,Nothing}

    indicesModel::Any

    store_result::Bool
    result::Any

    scalarLoss::Bool

    function FMU2EvaluationBatchElement(; scalarLoss::Bool = true, store_result::Bool=false)
        inst = new()

        inst.tStart = -Inf
        inst.tStop = Inf

        inst.loss = FMULoss(Inf)
        inst.losses = Array{FMULoss,1}()
        inst.step = 0

        inst.saveat = nothing
        inst.features = nothing
        inst.targets = nothing

        inst.indicesModel = nothing
        inst.result = nothing
        inst.scalarLoss = scalarLoss
        inst.store_result = store_result

        return inst
    end
end

function pasteFMUState!(fmu::FMU2, batchElement::FMU2SolutionBatchElement)
    c = getCurrentInstance(fmu)
    FMIBase.apply!(c, batchElement.snapshot)
    @info "Pasting snapshot @$(batchElement.snapshot.t)"
    return nothing
end

# make a copy of the curren time instant for the batch element
function copyFMUState!(fmu::FMU2, batchElement::FMU2SolutionBatchElement)
    c = getCurrentInstance(fmu)
    if isnothing(batchElement.snapshot)
        batchElement.snapshot = FMIBase.snapshot!(c)
        #batchElement.snapshot.t = batchElement.tStart
        @debug "New snapshot @$(batchElement.snapshot.t)"
    else
        tBefore = batchElement.snapshot.t
        FMIBase.update!(c, batchElement.snapshot)
        #batchElement.snapshot.t = batchElement.tStart
        tAfter = batchElement.snapshot.t

        # [Note] for non-consecutive batch elements (time gaps inside batch),
        #        it might be necessary to correct the new snapshot time to fit the old one.
        if tBefore != tAfter
            #batchElement.snapshot.t = max(tBefore, tAfter)
            #logInfo(fmu, "Corrected snapshot time from $(tAfter) to $(tBefore)")
            #logWarning(fmu, "Need to correct snapshot time from $(tAfter) to $(tBefore)")
        end

        @debug "Updated snapshot @$(batchElement.snapshot.t)"
    end
    return nothing
end

function run!(
    neuralFMU::ME_NeuralFMU,
    batchElement::FMU2SolutionBatchElement;
    nextBatchElement = nothing,
    kwargs...,
)

    neuralFMU.customCallbacksAfter = []
    neuralFMU.customCallbacksBefore = []

    function finishedCallback(x, t, integrator)
        c = getCurrentInstance(neuralFMU.fmu)

        t_set = unsense(t)
        x_set = unsense(x)

        #@info "PRE:  c.default_t=$(c.default_t) | c.t=$(c.t) | t=$(t_set)"

        # DifferentialEquations does not evaluate f(x, t) at t = t_f (of course)
        # however, we expect to make a snapshot at the very last state x(t_f),
        # therefore we need to manually set the FMU to this state with an additional evaluation.
        # Further, this evaluation does not need to be sensitive, and is only
        # to capture the correct FMUState.

        #@warn "remove this line"
        #c.default_t = t_set
        
        evaluateModel(neuralFMU, c, x_set; t=t_set)

        #@info "POST: c.default_t=$(c.default_t) | c.t=$(c.t) | t=$(t_set)"

        if c.default_t != c.t
            @warn "After neural FMU evaluation, default_t was not set, c.t=$(c.t) != c.default_t $(c.default_t)"
        end

        if t_set != c.t
            @warn "functionCallingCallback called for t=$(t_set) != FMU time $(c.t)"
        end
        if t_set != batchElement.tStop
            @warn "functionCallingCallback called for t=$(t_set) != batch element stop $(batchElement.tStop)"
        end
        if !isnothing(nextBatchElement) && t_set != nextBatchElement.tStart
            @warn "functionCallingCallback called for t=$(t_set) != next batch element start $(nextBatchElement.tStart)"
        end

        copyFMUState!(neuralFMU.fmu, nextBatchElement)
        return nothing
    end

    # STOP CALLBACK
    if !isnothing(nextBatchElement)
        stopcb = FunctionCallingCallback(
            finishedCallback;
            funcat = [batchElement.tStop],
        )
        push!(neuralFMU.customCallbacksAfter, stopcb)
    end

    writeSnapshot = nothing
    readSnapshot = nothing

    # on first run of the element, there is no snapshot
    if isnothing(batchElement.snapshot)
        c = getCurrentInstance(neuralFMU.fmu)
        batchElement.snapshot = snapshot!(c)
        writeSnapshot = batchElement.snapshot # needs to be updated, therefore write

        # to prevent logging a warning, because we will overwrite this soon
        writeSnapshot.t = batchElement.tStart 
    else
        readSnapshot = batchElement.snapshot
    end

    @debug "Running $(batchElement.tStart) with snapshot: $(!isnothing(batchElement.snapshot))..."

    solution = neuralFMU(
        batchElement.xStart,
        (batchElement.tStart, batchElement.tStop);
        readSnapshot = readSnapshot,
        writeSnapshot = writeSnapshot,
        saveat = batchElement.saveat,
        kwargs...,
    )

    if batchElement.store_result
        batchElement.result = solution
    end

    # @assert solution.states.t == batchElement.saveat "Batch element simulation failed, missmatch between `states.t` and `saveat`."

    neuralFMU.customCallbacksBefore = []
    neuralFMU.customCallbacksAfter = []

    batchElement.step += 1

    return solution
end

function run!(model, batchElement::FMU2EvaluationBatchElement, p = nothing)
    result = nothing 

    if isnothing(p) # implicite parameter model
        result =
            collect(model(f)[batchElement.indicesModel] for f in batchElement.features)
    else # explicite parameter model
        result =
            collect(model(p)(f)[batchElement.indicesModel] for f in batchElement.features)
    end

    if batchElement.store_result
        batchElement.result = result 
    end

    return result
end

function loss!(batchElement::FMU2SolutionBatchElement, lossFct, solution::FMUSolution; logLoss::Bool = false)

    loss = 0.0 # will be incremented

    if hasmethod(lossFct, Tuple{FMUSolution})
        loss = lossFct(solution)

    elseif hasmethod(lossFct, Tuple{FMUSolution, AbstractVector})
        loss = lossFct(solution, batchElement.targets)

    else # hasmethod(lossFct, Tuple{AbstractVector, AbstractVector})

        if solution.success
            if batchElement.scalarLoss
                for i = 1:length(batchElement.indicesModel)
                    dataTarget = collect(d[i] for d in batchElement.targets)
                    modelOutput = collect(
                        u[batchElement.indicesModel[i]] for
                        u in solution.states.u
                    )

                    loss += lossFct(modelOutput, dataTarget)
                end
            else
                dataTarget = batchElement.targets
                modelOutput = collect(
                    u[batchElement.indicesModel] for u in solution.states.u
                )

                loss = lossFct(modelOutput, dataTarget)
            end
        else
            @warn "Can't compute loss for batch element, because solution is invalid (`success=false`) for batch element\n$(batchElement)."
        end

    end

    batchElement.loss.step = batchElement.step
    batchElement.loss.time = time()
    batchElement.loss.loss = unsense(loss)

    ignore_derivatives() do
        if logLoss
            push!(batchElement.losses, deepcopy(batchElement.loss))
        end
    end

    return loss
end

function loss!(batchElement::FMU2EvaluationBatchElement, lossFct, result; logLoss::Bool = true)

    loss = 0.0 #  will be incremented 

    if batchElement.scalarLoss
        for i = 1:length(batchElement.indicesModel)
            dataTarget = collect(d[i] for d in batchElement.targets)
            modelOutput = collect(u[i] for u in result)

            loss += lossFct(modelOutput, dataTarget)
        end
    else
        dataTarget = batchElement.targets
        modelOutput = result

        loss = lossFct(modelOutput, dataTarget)
    end

    batchElement.loss.step = batchElement.step
    batchElement.loss.time = time()
    batchElement.loss.loss = unsense(loss)

    ignore_derivatives() do
        if logLoss
            push!(batchElement.losses, deepcopy(batchElement.loss))
        end
    end

    return loss
end

function _batchDataSolution!(
    batch::AbstractArray{<:FMIFlux.FMU2SolutionBatchElement},
    neuralFMU::NeuralFMU,
    x0_fun,
    train_t::AbstractArray{<:AbstractArray{<:Real}},
    targets::AbstractArray;
    kwargs...,
)

    len = length(train_t)
    for i = 1:len
        _batchDataSolution!(batch, neuralFMU, x0_fun, train_t[i], targets[i]; kwargs...)
    end
    return nothing
end

function _batchDataSolution!(
    batch::AbstractArray{<:FMIFlux.FMU2SolutionBatchElement},
    neuralFMU::NeuralFMU,
    x0_fun,
    train_t::AbstractArray{<:Real},
    targets::AbstractArray;
    batchDuration::Real = (train_t[end] - train_t[1]),
    indicesModel = 1:length(targets[1]),
    plot::Bool = false,
    scalarLoss::Bool = true,
)

    @assert length(train_t) == length(targets) "Timepoints in `train_t` ($(length(train_t))) must match number of `targets` ($(length(targets)))"

    canGetSetState = canGetSetFMUState(neuralFMU.fmu)
    if !canGetSetState
        logWarning(
            neuralFMU.fmu,
            "This FMU can't set/get a FMU state. This is suboptimal for batched training.",
        )
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

    numElements = floor(Integer, (train_t[end] - train_t[1]) / batchDuration)

    D = eltype(neuralFMU.fmu.modelDescription.discreteStateValueReferences)

    for i = 1:numElements

        element = FMIFlux.FMU2SolutionBatchElement{D}(; scalarLoss = scalarLoss)

        iStart = FMIFlux.timeToIndex(train_t, tStart + (i - 1) * batchDuration)
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

function batchDataSolution(
    neuralFMU::NeuralFMU,
    x0_fun,
    train_t,
    targets;
    batchDuration::Real = (train_t[end] - train_t[1]),
    indicesModel = 1:length(targets[1]),
    plot::Bool = false,
    scalarLoss::Bool = true,
    restartAtJump::Bool = true,
    solverKwargs...,
)

    batch = Array{FMIFlux.FMU2SolutionBatchElement,1}()
    _batchDataSolution!(
        batch,
        neuralFMU,
        x0_fun,
        train_t,
        targets;
        batchDuration = batchDuration,
        indicesModel = indicesModel,
        plot = plot,
        scalarLoss = scalarLoss,
    )

    numElements = length(batch)
    for i = 1:numElements

        nextBatchElement = nothing
        if i < numElements && batch[i].tStop == batch[i+1].tStart
            nextBatchElement = batch[i+1]
        end

        result = FMIFlux.run!(
            neuralFMU,
            batch[i];
            nextBatchElement = nextBatchElement,
            solverKwargs...,
        )

        if plot
            fig = FMIFlux.plot(batch[i], result)
            display(fig)
        end
    end

    return batch
end

function batchDataEvaluation(
    train_t::AbstractArray{<:Real},
    targets::AbstractArray,
    features::Union{AbstractArray,Nothing} = nothing;
    batchDuration::Real = (train_t[end] - train_t[1]),
    indicesModel = 1:length(targets[1]),
    plot::Bool = false,
    round_digits = 3,
    scalarLoss::Bool = true,
)

    batch = Array{FMIFlux.FMU2EvaluationBatchElement,1}()

    indicesData = 1:1

    tStart = train_t[1]

    iStart = timeToIndex(train_t, tStart)
    iStop = timeToIndex(train_t, tStart + batchDuration)

    startElement = FMIFlux.FMU2EvaluationBatchElement(; scalarLoss = scalarLoss)
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

    for i = 2:floor(Integer, (train_t[end] - train_t[1]) / batchDuration)
        push!(batch, FMIFlux.FMU2EvaluationBatchElement(; scalarLoss = scalarLoss))

        iStart = timeToIndex(train_t, tStart + (i - 1) * batchDuration)
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
