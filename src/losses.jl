#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

function loss(model, features::AbstractArray, targets::AbstractArray; 
    batchLen::Integer=length(features), batchIndex::Integer=rand(1:ceil(Integer, length(features)/batchLen)), target_model_range=nothing, target_data_range=1:length(targets[1]), lossFct=Flux.Losses.mse)

    @assert length(features) == length(targets) "Length of `features` and `targets` must be the same."

    # cut out data batch from data
    batch_begin = max(1 + (batchIndex-1) * batchLen, 1)
    batch_end = min(batchIndex * batchLen, length(features))
    features_data = features[batch_begin:batch_end]
    targets_data = targets[batch_begin:batch_end]

    # evaluate model
    targets_model = collect(model(f)[target_model_range] for f in features_data)
    targets_data = collect(td[target_data_range] for td in targets_data)

    if target_model_range == nothing 
        target_model_range = length(targets_model[1])
    end

    return lossFct(targets_model, targets_data)
end

function loss_NIPT(model, data::AbstractArray; kwargs...)
    return loss(model, data, data; kwargs...)
end
