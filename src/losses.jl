#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module Losses

using Flux
import ..FMIFlux: FMU2BatchElement, NeuralFMU, loss!, run!, ME_NeuralFMU

mse = Flux.Losses.mse
mae = Flux.Losses.mae

function last_element_rel(fun, a::AbstractArray, b::AbstractArray, lastElementRatio::Real)
    return (1.0-lastElementRatio) * fun(a[1:end-1], b[1:end-1]) + 
                lastElementRatio  * fun(a[  end  ], b[  end  ])
end

function mse_last_element_rel(a::AbstractArray, b::AbstractArray, lastElementRatio::Real=0.25)
    return last_element_rel(mse, a, b, lastElementRatio)
end

function mae_last_element_rel(a::AbstractArray, b::AbstractArray, lastElementRatio::Real=0.25)
    return last_element_rel(mae, a, b, lastElementRatio)
end

function mse_last_element(a::AbstractArray, b::AbstractArray)
    return mse(a[end], b[end])
end

function mae_last_element(a::AbstractArray, b::AbstractArray)
    return mae(a[end], b[end])
end

function loss(model, batchElement::FMU2BatchElement; 
    logLoss::Bool=true,
    lossFct=Flux.Losses.mse)

    model = nfmu.neuralODE.model[layers]

    loss = 0.0

    # evaluate model
    result = run!(model, batchElement)

    # for i in 1:length(batchElement.targets[1])
    #     targets_model = collect(r[batchElement.indicesModel[i]] for r in batchElement.result)
    #     targets_data = collect(td[i] for td in batchElement.targets)

    #     loss += lossFct(targets_model, targets_data)
    # end

    loss = loss!(batchElement, lossFct; logLoss=logLoss)

    return loss
end

function loss(nfmu::NeuralFMU, batch::AbstractArray{<:FMU2BatchElement}; 
    batchIndex::Integer=rand(1:length(batch)), 
    lossFct=Flux.Losses.mse,
    logLoss::Bool=true,
    kwargs...)

    # cut out data batch from data
    targets_data = batch[batchIndex].targets

    lastBatchElement = nothing 
    if batchIndex < length(batch)
        lastBatchElement = batch[batchIndex+1]
    end

    solution = run!(nfmu, batch[batchIndex]; lastBatchElement=lastBatchElement, progressDescr="Sim. Batch $(batchIndex)/$(length(batch)) |", kwargs...)
    
    if solution.success
        return loss!(batch[batchIndex], lossFct; logLoss=logLoss)
    else 
        @warn "Solving the NeuralFMU as part of the loss function failed. This is often because the ODE cannot be solved. Did you initialize the NeuralFMU model? Often additional solver errors/warnings are printed before this warning."
        return Inf
    end

end

function loss(model, batch::AbstractArray{<:FMU2BatchElement}; 
    batchIndex::Integer=rand(1:length(batch)), 
    lossFct=Flux.Losses.mse,
    logLoss::Bool=true)

    run!(model, batch[batchIndex])

    loss = loss!(batch[batchIndex], lossFct; logLoss=logLoss)

    return loss
end

function batch_loss(neuralFMU::ME_NeuralFMU, batch::AbstractArray{<:FMU2BatchElement}; update::Bool=false, logLoss::Bool=false, lossFct=nothing, kwargs...)

    accu = 0.0

    if update 
        @assert lossFct != nothing "update=true, but no keyword lossFct provided. Please provide one."
        numBatch = length(batch)
        for i in 1:numBatch
            b = batch[i]
            b_next = nothing 

            if i < numBatch
                b_next = batch[i+1]
            end

            if b.xStart != nothing
                run!(neuralFMU, b; lastBatchElement=b_next, progressDescr="Sim. Batch $(i)/$(numBatch) |", kwargs...)
            end
            
            l = loss!(b, lossFct; logLoss=logLoss)

            accu += l
        end
    else
        for b in batch

            @assert length(b.losses) > 0 "batch_loss(): `update=false` but no existing losses for batch element $(b)"
            accu += b.losses[end].loss
        end
    end

    return accu
end

function batch_loss(model, batch::AbstractArray{<:FMU2BatchElement}; update::Bool=false, logLoss::Bool=false, lossFct=nothing)

    accu = 0.0

    if update 
        @assert lossFct != nothing "update=true, but no keyword lossFct provided. Please provide one."
        numBatch = length(batch)
        for i in 1:numBatch
            b = batch[i]
            
            run!(model, b)
            
            accu += loss!(b, lossFct; logLoss=logLoss)
        end

    else

        for b in batch
            accu += b.losses[end].loss
        end

    end

    return accu
end

end # module
