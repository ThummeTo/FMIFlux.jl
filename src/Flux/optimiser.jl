#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

## Flux.Optimise ###

import FMIFlux: AbstractOptimiser

struct FluxOptimiserWrapper{G} <: AbstractOptimiser
    optim::Flux.Optimise.AbstractOptimiser
    grad_fun!::G
    grad_buffer::Union{AbstractVector{Float64},AbstractMatrix{Float64}}
    multiGrad::Bool

    function FluxOptimiserWrapper(
        optim::Flux.Optimise.AbstractOptimiser,
        grad_fun!::G,
        grad_buffer::AbstractVector{Float64},
    ) where {G}
        return new{G}(optim, grad_fun!, grad_buffer, false)
    end

    function FluxOptimiserWrapper(
        optim::Flux.Optimise.AbstractOptimiser,
        grad_fun!::G,
        grad_buffer::AbstractMatrix{Float64},
    ) where {G}
        return new{G}(optim, grad_fun!, grad_buffer, true)
    end

end
export FluxOptimiserWrapper

function FMIFlux.apply!(optim::FluxOptimiserWrapper, params)

    optim.grad_fun!(optim.grad_buffer, params)

    if optim.multiGrad
        return collect(
            Flux.Optimise.apply!(optim.optim, params, optim.grad_buffer[:, i]) for
            i = 1:size(optim.grad_buffer)[2]
        )
    else
        step = Flux.Optimise.apply!(optim.optim, params, optim.grad_buffer)
        return step
    end
end

# Dispatch for Flux.jl [Flux.Optimise.AbstractOptimiser]
function FMIFlux.train!(
    loss,
    neuralFMU::NeuralFMU,
    params, #::Union{Flux.Params,Zygote.Params,AbstractVector{<:AbstractVector{<:Real}}},
    data,
    optim::Flux.Optimise.AbstractOptimiser;
    gradient::Symbol = :ReverseDiff,
    chunk_size::Union{Integer,Symbol} = :auto_fmiflux,
    multiObjective::Bool = false,
    kwargs...,
)

    grad_buffer = nothing

    if multiObjective
        dim = loss(params[1])

        grad_buffer = zeros(Float64, length(params), length(dim))
    else
        grad_buffer = zeros(Float64, length(params))
    end

    grad_fun! = (G, p) -> FMIFlux.computeGradient!(G, loss, p, gradient, chunk_size, multiObjective)
    _optim = FluxOptimiserWrapper(optim, grad_fun!, grad_buffer)
    FMIFlux.train!(
        loss,
        neuralFMU,
        params,
        data,
        _optim;
        gradient = gradient,
        chunk_size = chunk_size,
        multiObjective = multiObjective,
        kwargs...,
    )
end

## Optimisers.AbstractRule ###

mutable struct OptimisersWrapper{G} <: AbstractOptimiser
    optim::Flux.Optimisers.AbstractRule
    grad_fun!::G
    grad_buffer::Union{AbstractVector{Float64},AbstractMatrix{Float64}}
    multiGrad::Bool
    state

    function OptimisersWrapper(
        optim::Flux.Optimisers.AbstractRule,
        grad_fun!::G,
        grad_buffer::AbstractVector{Float64},
        params
    ) where {G}
        state = Flux.Optimisers.setup(optim, params)
        return new{G}(optim, grad_fun!, grad_buffer, false, state)
    end

end
export OptimisersWrapper

function FMIFlux.apply!(optim::OptimisersWrapper, params)

    optim.grad_fun!(optim.grad_buffer, params)

    if optim.multiGrad
        return collect(
            Flux.Optimise.apply!(optim.optim, params, optim.grad_buffer[:, i]) for
            i = 1:size(optim.grad_buffer)[2]
        )
    else
        optim.state, new_ps = Flux.Optimisers.update!(optim.state, params, optim.grad_buffer)
        step = params .- new_ps
        @info "Grad: $(optim.grad_buffer[1:5])\nStep: $(step[1:5])\nParams: $(params[1:5])"
        return step
    end
end

# Dispatch for Flux.jl [Flux.Optimise.AbstractOptimiser]
function FMIFlux.train!(
    loss,
    neuralFMU::NeuralFMU,
    params, #::Union{Flux.Params,Zygote.Params,AbstractVector{<:AbstractVector{<:Real}}},
    data,
    optim::Flux.Optimisers.AbstractRule;
    gradient::Symbol = :ReverseDiff,
    chunk_size::Union{Integer,Symbol} = :auto_fmiflux,
    multiObjective::Bool = false,
    kwargs...,
)

    grad_buffer = zeros(Float64, length(params))
   
    grad_fun! = (G, p) -> FMIFlux.computeGradient!(G, loss, p, gradient, chunk_size, multiObjective)

    _optim = OptimisersWrapper(optim, grad_fun!, grad_buffer, params)
    FMIFlux.train!(
        loss,
        neuralFMU,
        params,
        data,
        _optim;
        gradient = gradient,
        chunk_size = chunk_size,
        multiObjective = multiObjective,
        kwargs...,
    )
end