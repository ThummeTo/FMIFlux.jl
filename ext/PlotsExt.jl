#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module PlotsExt 

import FMIFlux
using FMIFlux: FMU2SolutionBatchElement, FMU2BatchElement, BatchScheduler
using FMIFlux: nominalLoss, unsense
import Plots 

function FMIFlux.plot(args...; kwargs...)
    Plots.plot(args...; kwargs...)
end

function Plots.plot(batchElement::FMU2SolutionBatchElement, solution=batchElement.result; targets::Bool = true, plotkwargs...)

    fig = Plots.plot(; xlabel = "t [s]", plotkwargs...) # , title="loss[$(batchElement.step)] = $(nominalLoss(batchElement.losses[end]))")
    for i = 1:length(batchElement.indicesModel)
        if !isnothing(solution)
            @assert solution.states.t == batchElement.saveat "Batch element plotting failed, missmatch between `states.t` and `saveat`."

            Plots.plot!(
                fig,
                solution.states.t,
                collect(
                    unsense(u[batchElement.indicesModel[i]]) for
                    u in solution.states.u
                ),
                label = "Simulation #$(i)",
            )
        end
        if targets
            Plots.plot!(
                fig,
                batchElement.saveat,
                collect(d[i] for d in batchElement.targets),
                label = "Targets #$(i)",
            )
        end
    end

    return fig
end

function Plots.plot(
    batchElement::FMU2BatchElement,
    result=batchElement.result;
    targets::Bool = true,
    features::Bool = true,
    plotkwargs...,
)

    fig = Plots.plot(; xlabel = "t [s]", plotkwargs...) # , title="loss[$(batchElement.step)] = $(nominalLoss(batchElement.losses[end]))")

    if batchElement.features != nothing && features
        for i = 1:length(batchElement.features[1])
            Plots.plot!(
                fig,
                batchElement.saveat,
                collect(d[i] for d in batchElement.features),
                style = :dash,
                label = "Features #$(i)",
            )
        end
    end

    for i = 1:length(batchElement.indicesModel)
        if result != nothing
            Plots.plot!(
                fig,
                batchElement.saveat,
                collect(ForwardDiff.value(u[i]) for u in result),
                label = "Evaluation #$(i)",
            )
        end
        if targets
            Plots.plot!(
                fig,
                batchElement.saveat,
                collect(d[i] for d in batchElement.targets),
                label = "Targets #$(i)",
            )
        end
    end

    return fig
end

function Plots.plot(
    batch::AbstractArray{<:FMU2BatchElement};
    plot_mean::Bool = true,
    plot_shadow::Bool = true,
    plotkwargs...,
)

    num = length(batch)

    xs = 1:num
    ys = collect((nominalLoss(b) != Inf ? nominalLoss(b) : 0.0) for b in batch)

    fig = Plots.plot(; xlabel = "Batch ID", ylabel = "Loss", plotkwargs...)

    if plot_shadow
        ys_shadow = collect(
            (length(b.losses) > 1 ? nominalLoss(b.losses[end-1]) : 0.0) for b in batch
        )

        Plots.bar!(
            fig,
            xs,
            ys_shadow;
            label = "Previous loss",
            color = :green,
            bar_width = 1.0,
        )
    end

    Plots.bar!(fig, xs, ys; label = "Current loss", color = :blue, bar_width = 0.5)

    if plot_mean
        avgsum = mean(ys)
        Plots.plot!(fig, [1, num], [avgsum, avgsum]; label = "mean")
    end

    return fig
end

function Plots.plot(scheduler::BatchScheduler, lastIndex::Integer)
    num = length(scheduler.batch)

    xs = 1:num
    ys = collect((nominalLoss(b) != Inf ? nominalLoss(b) : 0.0) for b in scheduler.batch)
    ys_shadow = collect(
        (length(b.losses) > 1 ? nominalLoss(b.losses[end-1]) : 1e-16) for
        b in scheduler.batch
    )

    title = "[$(scheduler.step)]"
    if hasfield(typeof(scheduler), :printMsg)
        title = title * " " * scheduler.printMsg
    end

    fig = Plots.plot(;
        layout = Plots.grid(2, 1),
        size = (480, 960),
        xlabel = "Batch ID",
        ylabel = "Loss",
        background_color_legend = Plots.colorant"rgba(255,255,255,0.5)",
        title = title,
    )

    if hasfield(typeof(scheduler), :lossAccu)
        normScale = max(ys..., ys_shadow...) / max(scheduler.lossAccu...)
        Plots.bar!(
            fig[1],
            xs,
            scheduler.lossAccu .* normScale,
            label = "Accum. loss (norm.)",
            color = :blue,
            bar_width = 1.0,
            alpha = 0.2,
        )
    end

    good = []
    bad = []

    for i = 1:num
        if ys[i] > ys_shadow[i]
            push!(bad, i)
        else
            push!(good, i)
        end
    end

    Plots.bar!(
        fig[1],
        xs[good],
        ys[good],
        label = "Loss (better)",
        color = :green,
        bar_width = 1.0,
    )
    Plots.bar!(
        fig[1],
        xs[bad],
        ys[bad],
        label = "Loss (worse)",
        color = :orange,
        bar_width = 1.0,
    )

    for i = 1:length(ys_shadow)
        Plots.plot!(
            fig[1],
            [xs[i] - 0.5, xs[i] + 0.5],
            [ys_shadow[i], ys_shadow[i]],
            label = (i == 1 ? "Last loss" : :none),
            linewidth = 2,
            color = :black,
        )
    end

    if lastIndex > 0
        Plots.plot!(
            fig[1],
            [lastIndex],
            [0.0],
            color = :pink,
            marker = :circle,
            label = "Current ID [$(lastIndex)]",
            markersize = 5.0,
        ) # current batch element
    end
    Plots.plot!(
        fig[1],
        [scheduler.elementIndex],
        [0.0],
        color = :pink,
        marker = :circle,
        label = "Next ID [$(scheduler.elementIndex)]",
        markersize = 3.0,
    ) # next batch element

    Plots.plot!(fig[2], 1:length(scheduler.losses), scheduler.losses; yaxis = :log)

    display(fig)
end

function FMIFlux.plotLoss(batchElement::FMU2BatchElement; xaxis::Symbol = :steps)

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

    fig = Plots.plot(ts, ls, xlabel = tlabel, ylabel = "Loss")

    return fig
end

end # PlotsExt