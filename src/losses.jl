#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module Losses

using Flux
import ..FMIFlux: FMU2BatchElement, NeuralFMU, loss!, run!, ME_NeuralFMU, FMU2Solution
import FMIImport: unsense

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

function mae_dev(a::AbstractArray, b::AbstractArray, dev::AbstractArray)
    num = length(a)
    Δ = abs.(a .- b)
    Δ -= abs.(dev)
    Δ = collect(max(val, 0.0) for val in Δ)
    Δ = sum(Δ) / num
    return Δ
end

function mse_dev(a::AbstractArray, b::AbstractArray, dev::AbstractArray)
    num = length(a)
    Δ = abs.(a .- b)
    Δ -= abs.(dev)
    Δ = collect(max(val, 0.0) for val in Δ)
    Δ = sum(Δ .^ 2) / num
    return Δ
end

function max_dev(a::AbstractArray, b::AbstractArray, dev::AbstractArray)
    Δ = abs.(a .- b)
    Δ -= abs.(dev)
    Δ = collect(max(val, 0.0) for val in Δ)
    Δ = max(Δ...)
    return Δ
end

function stiffness_corridor(solution::FMU2Solution, corridor::AbstractArray{<:AbstractArray{<:Tuple{Real, Real}}}; lossFct=Flux.Losses.mse)
    @assert !isnothing(solution.eigenvalues) "stiffness_corridor: Need eigenvalue information, that is not present in the given `FMU2Solution`. Use keyword `recordEigenvalues=true` for FMU or NeuralFMU simulation."

    eigs_over_time = solution.eigenvalues.saveval 
    num_eigs_over_time = length(eigs_over_time)
    
    @assert num_eigs_over_time == length(corridor) "stiffness_corridor: length of time points with eigenvalues $(num_eigs_over_time) doesn't match time points in corridor $(length(corridor))."

    l = 0.0
    for i in 2:num_eigs_over_time
        eigs = eigs_over_time[i]
        num_eigs = Int(length(eigs)/2)

        for j in 1:num_eigs
            re = eigs[(j-1)*2+1]
            im = eigs[j*2]

            c_min, c_max = corridor[i][j]
            if re > c_max 
                l += lossFct(re, c_max) / num_eigs / num_eigs_over_time
            end
            if re < c_min 
                l += lossFct(c_min, re) / num_eigs / num_eigs_over_time
            end
        end
    end

    return l
end

function stiffness_corridor(solution::FMU2Solution, corridor::AbstractArray{<:Tuple{Real, Real}}; lossFct=Flux.Losses.mse)
    @assert !isnothing(solution.eigenvalues) "stiffness_corridor: Need eigenvalue information, that is not present in the given `FMU2Solution`. Use keyword `recordEigenvalues=true` for FMU or NeuralFMU simulation."

    eigs_over_time = solution.eigenvalues.saveval 
    num_eigs_over_time = length(eigs_over_time)
    
    @assert num_eigs_over_time == length(corridor) "stiffness_corridor: length of time points with eigenvalues $(num_eigs_over_time) doesn't match time points in corridor $(length(corridor))."

    l = 0.0
    for i in 2:num_eigs_over_time
        eigs = eigs_over_time[i]
        num_eigs = Int(length(eigs)/2)
        c_min, c_max = corridor[i]

        for j in 1:num_eigs
            re = eigs[(j-1)*2+1]
            im = eigs[j*2]

            if re > c_max 
                l += lossFct(re, c_max) / num_eigs / num_eigs_over_time
            end
            if re < c_min 
                l += lossFct(c_min, re) / num_eigs / num_eigs_over_time
            end
        end
    end

    return l
end

function stiffness_corridor(solution::FMU2Solution, corridor::Tuple{Real, Real}; lossFct=Flux.Losses.mse)
    @assert !isnothing(solution.eigenvalues) "stiffness_corridor: Need eigenvalue information, that is not present in the given `FMU2Solution`. Use keyword `recordEigenvalues=true` for FMU or NeuralFMU simulation."

    eigs_over_time = solution.eigenvalues.saveval 
    num_eigs_over_time = length(eigs_over_time)
    
    c_min, c_max = corridor

    l = 0.0
    for i in 2:num_eigs_over_time
        eigs = eigs_over_time[i]
        num_eigs = Int(length(eigs)/2)

        for j in 1:num_eigs
            re = eigs[(j-1)*2+1]
            im = eigs[j*2]

            if re > c_max 
                l += lossFct(re, c_max) / num_eigs / num_eigs_over_time
            end
            if re < c_min 
                l += lossFct(re, c_min) / num_eigs / num_eigs_over_time
            end
        end
    end

    return l
end

function loss(model, batchElement::FMU2BatchElement; 
    logLoss::Bool=true,
    lossFct=Flux.Losses.mse, p=nothing)

    model = nfmu.neuralODE.model[layers]

    # evaluate model
    result = run!(model, batchElement, p=p)

    return loss!(batchElement, lossFct; logLoss=logLoss)
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
    
    if !solution.success
        @warn "Solving the NeuralFMU as part of the loss function failed with return code `$(solution.states.retcode)`.\nThis is often because the ODE cannot be solved. Did you initialize the NeuralFMU model?\nOften additional solver errors/warnings are printed before this warning.\nHowever, it is tried to compute a loss on the partial retrieved solution from $(unsense(solution.states.t[1]))s to $(unsense(solution.states.t[end]))s."
    end 
        
    return loss!(batch[batchIndex], lossFct; logLoss=logLoss)
end

function loss(model, batch::AbstractArray{<:FMU2BatchElement}; 
    batchIndex::Integer=rand(1:length(batch)), 
    lossFct=Flux.Losses.mse,
    logLoss::Bool=true, p=nothing)

    run!(model, batch[batchIndex], p)

    return loss!(batch[batchIndex], lossFct; logLoss=logLoss)
end

function batch_loss(neuralFMU::ME_NeuralFMU, batch::AbstractArray{<:FMU2BatchElement}; update::Bool=false, logLoss::Bool=false, lossFct=nothing, kwargs...)

    accu = nothing

    if update 
        @assert lossFct != nothing "update=true, but no keyword lossFct provided. Please provide one."
        numBatch = length(batch)
        for i in 1:numBatch
            b = batch[i]

            b_next = nothing 
            if i < numBatch
                b_next = batch[i+1]
            end

            if !isnothing(b.xStart)
                run!(neuralFMU, b; lastBatchElement=b_next, progressDescr="Sim. Batch $(i)/$(numBatch) |", kwargs...)
            end
            
            if isnothing(accu)
                accu = loss!(b, lossFct; logLoss=logLoss)
            else
                accu += loss!(b, lossFct; logLoss=logLoss)
            end
           
        end
    else
        for b in batch

            @assert length(b.losses) > 0 "batch_loss(): `update=false` but no existing losses for batch element $(b)"

            if isnothing(accu)
                accu = b.losses[end].loss 
            else
                accu += b.losses[end].loss 
            end
        end
    end

    return accu
end

function batch_loss(model, batch::AbstractArray{<:FMU2BatchElement}; update::Bool=false, logLoss::Bool=false, lossFct=nothing, p=nothing)

    accu = nothing

    if update 
        @assert lossFct != nothing "update=true, but no keyword lossFct provided. Please provide one."
        numBatch = length(batch)
        for i in 1:numBatch
            b = batch[i]
            
            run!(model, b, p)
            
            if isnothing(accu)
                accu = loss!(b, lossFct; logLoss=logLoss)
            else
                accu += loss!(b, lossFct; logLoss=logLoss)
            end
        end

    else

        for b in batch
            if isnothing(accu)
                accu = nominalLoss(b.losses[end])
            else
                accu += nominalLoss(b.losses[end])
            end
        end

    end

    return accu
end

mutable struct ToggleLoss
    index::Int 
    losses

    function ToggleLoss(losses...)
        @assert length(losses) >= 2 "ToggleLoss needs at least 2 losses, $(length(losses)) given."
        return new(1, losses)
    end
end

function (t::ToggleLoss)(args...; kwargs...)
    ret = t.losses[t.index](args...; kwargs...)
    t.index += 1
    if t.index > length(t.losses)
        t.index = 1 
    end
    return ret 
end

end # module
