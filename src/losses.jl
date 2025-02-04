#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module Losses

import ..FMIFlux: FMU2BatchElement, NeuralFMU, loss!, run!, ME_NeuralFMU, FMUSolution
import ..FMIFlux.FMIImport.FMIBase: unsense, logWarning

function mean_error_sum(a, b, fun)
    sum = 0.0
    len_a = length(a)
    len_b = length(b)
    len = len_a

    if len_a != len_b
        len = min(len_a, len_b)
        @warn "Length a ($(len_a)) != Length b ($(len_b)), not entire set is compared, only the first $(len) elements!"
    end 

    for i in 1:len 
        sum += fun(a[i], b[i])
    end
    
    return sum / len
end

#mse = Flux.Losses.mse
function mse(a, b)
    fun = function(x, y)
        (x .- y) * (x .- y)
    end
    return mean_error_sum(a, b, fun)
end

#mae = Flux.Losses.mae
function mae(a, b)
    fun = function(x, y)
        abs(x .- y)
    end
    return mean_error_sum(a, b, fun)
end

function last_element_rel(fun, a::AbstractArray, b::AbstractArray, lastElementRatio::Real)
    return (1.0 - lastElementRatio) * fun(a[1:end-1], b[1:end-1]) +
           lastElementRatio * fun(a[end], b[end])
end

function mse_last_element_rel(
    a::AbstractArray,
    b::AbstractArray,
    lastElementRatio::Real = 0.25,
)
    return last_element_rel(mse, a, b, lastElementRatio)
end

function mae_last_element_rel(
    a::AbstractArray,
    b::AbstractArray,
    lastElementRatio::Real = 0.25,
)
    return last_element_rel(mae, a, b, lastElementRatio)
end

function mse_last_element(a::AbstractArray, b::AbstractArray)
    return mse(a[end], b[end])
end

function mae_last_element(a::AbstractArray, b::AbstractArray)
    return mae(a[end], b[end])
end

function deviation(a::AbstractArray, b::AbstractArray, dev::AbstractArray)
    Δ = abs.(a .- b)
    Δ -= abs.(dev)
    Δ = collect(max(val, 0.0) for val in Δ)

    return Δ
end

function mae_dev(a::AbstractArray, b::AbstractArray, dev::AbstractArray)
    num = length(a)
    Δ = deviation(a, b, dev)
    Δ = sum(Δ) / num
    return Δ
end

function mse_dev(a::AbstractArray, b::AbstractArray, dev::AbstractArray)
    num = length(a)
    Δ = deviation(a, b, dev)
    Δ = sum(Δ .^ 2) / num
    return Δ
end

function max_dev(a::AbstractArray, b::AbstractArray, dev::AbstractArray)
    Δ = deviation(a, b, dev)
    Δ = max(Δ...)
    return Δ
end

function mae_last_element_rel_dev(
    a::AbstractArray,
    b::AbstractArray,
    dev::AbstractArray,
    lastElementRatio::Real,
)
    num = length(a)
    Δ = deviation(a, b, dev)
    Δ[1:end-1] .*= (1.0 - lastElementRatio)
    Δ[end] *= lastElementRatio
    Δ = sum(Δ) / num
    return Δ
end

function mse_last_element_rel_dev(
    a::AbstractArray,
    b::AbstractArray,
    dev::AbstractArray,
    lastElementRatio::Real,
)
    num = length(a)
    Δ = deviation(a, b, dev)
    Δ = Δ .^ 2
    Δ[1:end-1] .*= (1.0 - lastElementRatio)
    Δ[end] *= lastElementRatio
    Δ = sum(Δ) / num
    return Δ
end

function stiffness_corridor(
    solution::FMUSolution,
    corridor::AbstractArray{<:AbstractArray{<:Tuple{Real,Real}}};
    lossFct = mse,
)
    @assert !isnothing(solution.eigenvalues) "stiffness_corridor: Need eigenvalue information, that is not present in the given `FMUSolution`. Use keyword `recordEigenvalues=true` for FMU or NeuralFMU simulation."

    eigs_over_time = solution.eigenvalues.saveval
    num_eigs_over_time = length(eigs_over_time)

    @assert num_eigs_over_time == length(corridor) "stiffness_corridor: length of time points with eigenvalues $(num_eigs_over_time) doesn't match time points in corridor $(length(corridor))."

    l = 0.0
    for i = 2:num_eigs_over_time
        eigs = eigs_over_time[i]
        num_eigs = Int(length(eigs) / 2)

        for j = 1:num_eigs
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

function stiffness_corridor(
    solution::FMUSolution,
    corridor::AbstractArray{<:Tuple{Real,Real}};
    lossFct = mse,
)
    @assert !isnothing(solution.eigenvalues) "stiffness_corridor: Need eigenvalue information, that is not present in the given `FMUSolution`. Use keyword `recordEigenvalues=true` for FMU or NeuralFMU simulation."

    eigs_over_time = solution.eigenvalues.saveval
    num_eigs_over_time = length(eigs_over_time)

    @assert num_eigs_over_time == length(corridor) "stiffness_corridor: length of time points with eigenvalues $(num_eigs_over_time) doesn't match time points in corridor $(length(corridor))."

    l = 0.0
    for i = 2:num_eigs_over_time
        eigs = eigs_over_time[i]
        num_eigs = Int(length(eigs) / 2)
        c_min, c_max = corridor[i]

        for j = 1:num_eigs
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

function stiffness_corridor(
    solution::FMUSolution,
    corridor::Tuple{Real,Real};
    lossFct = mse,
)
    @assert !isnothing(solution.eigenvalues) "stiffness_corridor: Need eigenvalue information, that is not present in the given `FMUSolution`. Use keyword `recordEigenvalues=true` for FMU or NeuralFMU simulation."

    eigs_over_time = solution.eigenvalues.saveval
    num_eigs_over_time = length(eigs_over_time)

    c_min, c_max = corridor

    l = 0.0
    for i = 2:num_eigs_over_time
        eigs = eigs_over_time[i]
        num_eigs = Int(length(eigs) / 2)

        for j = 1:num_eigs
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

function loss(
    model,
    batchElement::FMU2BatchElement;
    logLoss::Bool = true,
    lossFct = mse,
    p = nothing,
)

    model = nfmu.neuralODE.model[layers]

    # evaluate model
    result = run!(model, batchElement, p = p)

    return loss!(batchElement, lossFct; logLoss = logLoss)
end

function loss(
    nfmu::NeuralFMU,
    batch::AbstractArray{<:FMU2BatchElement};
    batchIndex::Integer = rand(1:length(batch)),
    lossFct = mse,
    logLoss::Bool = true,
    solvekwargs...,
)

    # cut out data batch from data
    targets_data = batch[batchIndex].targets

    nextBatchElement = nothing
    if batchIndex < length(batch) && batch[batchIndex].tStop == batch[batchIndex+1].tStart
        nextBatchElement = batch[batchIndex+1]
    end

    solution = run!(
        nfmu,
        batch[batchIndex];
        nextBatchElement = nextBatchElement,
        progressDescr = "Sim. Batch $(batchIndex)/$(length(batch)) |",
        solvekwargs...,
    )

    if !solution.success
        logWarning(
            nfmu.fmu,
            "Solving the NeuralFMU as part of the loss function failed with return code `$(solution.states.retcode)`.\nThis is often because the ODE cannot be solved. Did you initialize the NeuralFMU model?\nOften additional solver errors/warnings are printed before this warning.\nHowever, it is tried to compute a loss on the partial retrieved solution from $(unsense(solution.states.t[1]))s to $(unsense(solution.states.t[end]))s.",
        )
        return Inf
    else
        return loss!(batch[batchIndex], lossFct; logLoss = logLoss)
    end
end

function loss(
    model,
    batch::AbstractArray{<:FMU2BatchElement};
    batchIndex::Integer = rand(1:length(batch)),
    lossFct = mse,
    logLoss::Bool = true,
    p = nothing,
)

    run!(model, batch[batchIndex], p)

    return loss!(batch[batchIndex], lossFct; logLoss = logLoss)
end

function batch_loss(
    neuralFMU::ME_NeuralFMU,
    batch::AbstractArray{<:FMU2BatchElement};
    update::Bool = false,
    logLoss::Bool = false,
    lossFct = nothing,
    kwargs...,
)

    accu = nothing

    if update
        @assert lossFct != nothing "update=true, but no keyword lossFct provided. Please provide one."
        numBatch = length(batch)
        for i = 1:numBatch
            b = batch[i]

            b_next = nothing
            if i < numBatch && batch[i].tStop == batch[i+1].tStart
                b_next = batch[i+1]
            end

            if !isnothing(b.xStart)
                run!(
                    neuralFMU,
                    b;
                    nextBatchElement = b_next,
                    progressDescr = "Sim. Batch $(i)/$(numBatch) |",
                    kwargs...,
                )
            end

            if isnothing(accu)
                accu = loss!(b, lossFct; logLoss = logLoss)
            else
                accu += loss!(b, lossFct; logLoss = logLoss)
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

function batch_loss(
    model,
    batch::AbstractArray{<:FMU2BatchElement};
    update::Bool = false,
    logLoss::Bool = false,
    lossFct = nothing,
    p = nothing,
)

    accu = nothing

    if update
        @assert lossFct != nothing "update=true, but no keyword lossFct provided. Please provide one."
        numBatch = length(batch)
        for i = 1:numBatch
            b = batch[i]

            run!(model, b, p)

            if isnothing(accu)
                accu = loss!(b, lossFct; logLoss = logLoss)
            else
                accu += loss!(b, lossFct; logLoss = logLoss)
            end
        end

    else

        for b in batch
            if isnothing(accu)
                accu = nominalLoss(b)
            else
                accu += nominalLoss(b)
            end
        end

    end

    return accu
end

mutable struct ToggleLoss
    index::Int
    losses::Any

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

"""
Compares non-equidistant (or equidistant) datapoints by linear interpolating and comparing at given interpolation points `t_comp`. 
(Zygote-friendly: Zygote can differentiate through via AD.)
"""
function mse_interpolate(t1, x1, t2, x2, t_comp)
    #lin1 = LinearInterpolation(t1, x1)
    #lin2 = LinearInterpolation(t2, x2)
    ar1 = collect(lin_interp(t1, x1, t_sample) for t_sample in t_comp) #lin1.(t_comp)
    ar2 = collect(lin_interp(t2, x2, t_sample) for t_sample in t_comp) #lin2.(t_comp)
    mse(ar1, ar2)
end

# Helper: simple linear interpolation 
function lin_interp(t, x, t_sample)
    if t_sample <= t[1]
        return x[1]
    end

    if t_sample >= t[end]
        return x[end]
    end

    i = 1
    while t_sample > t[i]
        i += 1
    end

    x_left = x[i-1]
    x_right = x[i]

    t_left = t[i-1]
    t_right = t[i]

    dx = x_right - x_left
    dt = t_right - t_left
    h = t_sample - t_left

    x_left + dx / dt * h
end

end # module
