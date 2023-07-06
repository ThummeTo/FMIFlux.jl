#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Printf
using Colors

abstract type BatchScheduler end

# ToDo: DocString update.
"""
    Computes all batch element losses. Picks the batch element with the greatest loss as next training element.
"""
mutable struct WorstElementScheduler <: BatchScheduler

    ### mandatory ###
    step::Integer
    elementIndex::Integer
    applyStep::Integer
    plotStep::Integer
    batch
    neuralFMU::NeuralFMU

    ### type specific ###
    lossFct
    runkwargs
    printMsg::String
    
    function WorstElementScheduler(neuralFMU::NeuralFMU, batch, lossFct=Flux.Losses.mse; applyStep::Integer=1, plotStep::Integer=1)
        inst = new()
        inst.neuralFMU = neuralFMU
        inst.step = 0
        inst.elementIndex = 0
        inst.batch = batch
        inst.lossFct = lossFct
        inst.applyStep = applyStep
        inst.plotStep = plotStep

        inst.printMsg = ""
       
        return inst
    end
end

# ToDo: DocString update.
"""
    Computes all batch element losses. Picks the batch element with the greatest accumulated loss as next training element. If picked, accumulated loss is resetted.
    (Prevents starvation of batch elements with little loss)
"""
mutable struct LossAccumulationScheduler <: BatchScheduler

    ### mandatory ###
    step::Integer
    elementIndex::Integer
    applyStep::Integer
    plotStep::Integer
    batch
    neuralFMU::NeuralFMU

    ### type specific ###
    lossFct
    runkwargs
    printMsg::String
    lossAccu::Array{<:Real}
    updateStep::Integer
    
    function LossAccumulationScheduler(neuralFMU::NeuralFMU, batch, lossFct=Flux.Losses.mse; applyStep::Integer=1, plotStep::Integer=1, updateStep::Integer=1)
        inst = new()
        inst.neuralFMU = neuralFMU
        inst.step = 0
        inst.elementIndex = 0
        inst.batch = batch
        inst.lossFct = lossFct
        inst.applyStep = applyStep
        inst.plotStep = plotStep
        inst.updateStep = updateStep
        
        inst.printMsg = ""
        inst.lossAccu = zeros(length(batch))

        return inst
    end
end

# ToDo: DocString update.
"""
    Computes all batch element losses. Picks the batch element with the greatest grow in loss (derivative) as next training element.
"""
mutable struct WorstGrowScheduler <: BatchScheduler

    ### mandatory ###
    step::Integer
    elementIndex::Integer
    applyStep::Integer
    plotStep::Integer
    batch
    neuralFMU::NeuralFMU

    ### type specific ###
    lossFct
    runkwargs
    printMsg::String

    function WorstGrowScheduler(neuralFMU::NeuralFMU, batch, lossFct=Flux.Losses.mse; applyStep::Integer=1, plotStep::Integer=1)
        inst = new()
        inst.neuralFMU = neuralFMU
        inst.step = 0
        inst.elementIndex = 0
        inst.batch = batch
        inst.lossFct = lossFct
        inst.applyStep = applyStep
        inst.plotStep = plotStep

        inst.printMsg = ""

        return inst
    end
end

# ToDo: DocString update.
"""
    Picks a random batch element as next training element.
"""
mutable struct RandomScheduler <: BatchScheduler

    ### mandatory ###
    step::Integer
    elementIndex::Integer
    applyStep::Integer
    plotStep::Integer
    batch
    neuralFMU::NeuralFMU
    
    ### type specific ###
    # none

    function RandomScheduler(neuralFMU::NeuralFMU, batch; applyStep::Integer=1, plotStep::Integer=1)
        inst = new()
        inst.neuralFMU = neuralFMU
        inst.step = 0
        inst.elementIndex = 0
        inst.batch = batch
        inst.applyStep = applyStep
        inst.plotStep = plotStep

        return inst
    end
end

# ToDo: DocString update.
"""
    Sequentially runs over all elements.
"""
mutable struct SequentialScheduler <: BatchScheduler

    ### mandatory ###
    step::Integer
    elementIndex::Integer
    applyStep::Integer
    plotStep::Integer
    batch
    neuralFMU::NeuralFMU
    
    ### type specific ###
    # none

    function SequentialScheduler(neuralFMU::NeuralFMU, batch; applyStep::Integer=1, plotStep::Integer=1)
        inst = new()
        inst.neuralFMU = neuralFMU
        inst.step = 0
        inst.elementIndex = 0
        inst.batch = batch
        inst.applyStep = applyStep
        inst.plotStep = plotStep

        return inst
    end
end

function initialize!(scheduler::BatchScheduler; runkwargs...)

    lastIndex = 0
    scheduler.step = 0
    scheduler.elementIndex = 0

    if hasfield(typeof(scheduler), :runkwargs)
        scheduler.runkwargs = runkwargs
    end

    scheduler.elementIndex = apply!(scheduler)
    
    if scheduler.plotStep > 0
        plot(scheduler, lastIndex)
    end
end

function update!(scheduler::BatchScheduler)

    lastIndex = scheduler.elementIndex

    scheduler.step += 1
    
    if scheduler.applyStep > 0 && scheduler.step % scheduler.applyStep == 0
        scheduler.elementIndex = apply!(scheduler)
    end

    if scheduler.plotStep > 0 && scheduler.step % scheduler.plotStep == 0
        plot(scheduler, lastIndex)
    end
end

function plot(scheduler::BatchScheduler, lastIndex::Integer)
    num = length(scheduler.batch)

    xs = 1:num 
    ys = collect((length(b.losses) > 0 ? b.losses[end].loss : 0.0) for b in scheduler.batch)
    ys_shadow = collect((length(b.losses) > 1 ? b.losses[end-1].loss : 1e-16) for b in scheduler.batch)
    
    title = "[$(scheduler.step)]" 
    if hasfield(typeof(scheduler), :printMsg)
        title = title * " " * scheduler.printMsg
    end

    fig = Plots.plot(; xlabel="Batch ID", ylabel="Loss", background_color_legend=colorant"rgba(255,255,255,0.5)", title=title)

    if hasfield(typeof(scheduler), :lossAccu)
        normScale = max(ys..., ys_shadow...) / max(scheduler.lossAccu...)
        Plots.bar!(fig, xs, scheduler.lossAccu .* normScale, label="Accum. loss (norm.)", color=:blue, bar_width=1.0, alpha=0.2);
    end

    good = []
    bad = []

    for i in 1:num 
        if ys[i] > ys_shadow[i]
            push!(bad, i)
        else
            push!(good, i)
        end
    end
    
    Plots.bar!(fig, xs[good], ys[good], label="Loss (better)", color=:green, bar_width=1.0);
    Plots.bar!(fig, xs[bad], ys[bad], label="Loss (worse)", color=:orange, bar_width=1.0);

    for i in 1:length(ys_shadow)
        Plots.plot!(fig, [xs[i]-0.5, xs[i]+0.5], [ys_shadow[i], ys_shadow[i]], label=(i == 1 ? "Last loss" : :none), linewidth=2, color=:black);
    end
    
    if lastIndex > 0
        Plots.plot!(fig, [lastIndex], [0.0], color=:pink, marker=:circle, label="Current ID [$(lastIndex)]", markersize = 5.0) # current batch element
    end
    Plots.plot!(fig, [scheduler.elementIndex], [0.0], color=:pink, marker=:circle, label="Next ID [$(scheduler.elementIndex)]", markersize = 3.0) # next batch element
    display(fig)
end

"""
    Rounds a given `number` to a string with a maximum length of `len`.
    Exponentials are used if suitable.
"""
function roundToLength(number::Real, len::Integer)

    @assert len >= 5 "`len` must be at least `5`."

    if number == 0.0
        return "0.0"
    end

    expLen = 0

    if abs(number) <= 1.0 
        expLen = Integer(floor(log10(1.0/number))) + 1
    else 
        expLen = Integer(floor(log10(number))) + 1
    end

    len -= 4 # spaces needed for "+" (or "-"), "e", "." and leading number
    if expLen >= 100
        len -= 3 # 3 spaces needed for large exponent
    else
        len -= 2 # 2 spaces needed for regular exponent
    end
    
    return Printf.format(Printf.Format("%.$(len)e"), number)
end

function apply!(scheduler::WorstElementScheduler; print::Bool=true)
    
    avgsum = 0.0
    losssum = 0.0

    maxe = 0.0
    maxind = 0

    num = length(scheduler.batch)
    for i in 1:num
        #l = scheduler.batch[i].losses[end].loss

        FMIFlux.run!(scheduler.neuralFMU, scheduler.batch[i]; scheduler.runkwargs...)
        l = FMIFlux.loss!(scheduler.batch[i], scheduler.lossFct; logLoss=true)
        
        losssum += l
        avgsum += l / num
        if l > maxe
            maxe = l 
            maxind = i
        end
    
    end

    if print
        scheduler.printMsg = "AVG: $(roundToLength(avgsum, 8)) | MAX: $(roundToLength(maxe, 8)) @ #$(scheduler.elementIndex)"
        @info scheduler.printMsg
    end

    return maxind
end

function apply!(scheduler::LossAccumulationScheduler; print::Bool=true)
    
    avgsum = 0.0
    losssum = 0.0

    maxe = 0.0
    nextind = 1

    # reset current accu loss
    scheduler.lossAccu[scheduler.elementIndex] = 0.0

    num = length(scheduler.batch)
    for i in 1:num

        l = 0.0

        if length(scheduler.batch[i].losses) >= 1
            l = scheduler.batch[i].losses[end].loss
        end
        
        if scheduler.step % scheduler.updateStep == 0
            FMIFlux.run!(scheduler.neuralFMU, scheduler.batch[i]; scheduler.runkwargs...)
            l = FMIFlux.loss!(scheduler.batch[i], scheduler.lossFct; logLoss=true)
        end

        scheduler.lossAccu[i] += l

        losssum += l
        avgsum += l / num

        if l > maxe
            maxe = l 
        end
    end

    # find largest accumulated loss
    for i in 1:num
        if scheduler.lossAccu[i] > scheduler.lossAccu[nextind]
            nextind = i
        end
    end

    if print
        scheduler.printMsg = "AVG: $(roundToLength(avgsum, 8)) | MAX: $(roundToLength(maxe, 8)) @ #$(scheduler.elementIndex)"
        @info scheduler.printMsg
    end

    return nextind
end

function apply!(scheduler::WorstGrowScheduler; print::Bool=true)
    
    avgsum = 0.0
    losssum = 0.0

    maxe = 0.0
    maxe_der = -Inf
    maxind = 0

    num = length(scheduler.batch)
    for i in 1:num
       
        FMIFlux.run!(scheduler.neuralFMU, scheduler.batch[i]; scheduler.runkwargs...)
        l = FMIFlux.loss!(scheduler.batch[i], scheduler.lossFct; logLoss=true)

        l_der = l # fallback for first run (greatest error)
        if length(scheduler.batch[i].losses) >= 2
            l_der = (l - scheduler.batch[i].losses[end-1].loss)
        end
        
        losssum += l
        avgsum += l / num

        if l > maxe 
            maxe = l 
        end

        if l_der > maxe_der
            maxe_der = l_der
            maxind = i
        end
    
    end

    if print
        scheduler.printMsg = "AVG: $(roundToLength(avgsum, 8)) | MAX: $(roundToLength(maxe, 8)) @ #$(scheduler.elementIndex)"
        @info scheduler.printMsg
    end

    return maxind
end

function apply!(scheduler::RandomScheduler; print::Bool=true)

    if print
        @info "$(scheduler.elementIndex) [$(scheduler.step)]"
    end

    return rand(1:length(scheduler.batch))
end

function apply!(scheduler::SequentialScheduler; print::Bool=true)

    if print
        @info "$(scheduler.elementIndex) [$(scheduler.step)]"
    end

    next = scheduler.elementIndex+1
    if next > length(scheduler.batch)
        next = 1
    end

    return next
end