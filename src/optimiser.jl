#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Flux
import Optim

abstract type AbstractOptimiser end

### Optim.jl ###

struct OptimOptimiserWrapper{G} <: AbstractOptimiser
    optim::Optim.AbstractOptimizer
    grad_fun!::G

    state::Union{Optim.AbstractOptimizerState, Nothing}
    d::Union{Optim.OnceDifferentiable, Nothing}
    options

    function OptimOptimiserWrapper(optim::Optim.AbstractOptimizer, grad_fun!::G, loss, params) where {G}
        options = Optim.Options(iterations=1)
        autodiff = :finite
        inplace = true

        d = Optim.promote_objtype(optim, params, autodiff, inplace, loss, grad_fun!)
        state = Optim.initial_state(optim, options, d, params)

        return new{G}(optim, grad_fun!, state, d, options)
    end

end
export OptimOptimiserWrapper
  
function apply!(optim::OptimOptimiserWrapper, params)

    res = Optim.optimize(optim.d, params, optim.optim, optim.options, optim.state)
    step = params .- Optim.minimizer(res)

    return step
end

### Flux.Optimisers ###

struct FluxOptimiserWrapper{G} <: AbstractOptimiser
    optim::Flux.Optimise.AbstractOptimiser
    grad_fun!::G
    grad_buffer::Union{AbstractVector{Float64}, AbstractMatrix{Float64}}
    multiGrad::Bool

    function FluxOptimiserWrapper(optim::Flux.Optimise.AbstractOptimiser, grad_fun!::G, grad_buffer::AbstractVector{Float64}) where {G}
        return new{G}(optim, grad_fun!, grad_buffer, false)
    end

    function FluxOptimiserWrapper(optim::Flux.Optimise.AbstractOptimiser, grad_fun!::G, grad_buffer::AbstractMatrix{Float64}) where {G}
        return new{G}(optim, grad_fun!, grad_buffer, true)
    end

end
export FluxOptimiserWrapper
  
function apply!(optim::FluxOptimiserWrapper, params)

    optim.grad_fun!(optim.grad_buffer, params)

    if optim.multiGrad
        return collect(Flux.Optimise.apply!(optim.optim, params, optim.grad_buffer[:,i]) for i in 1:size(optim.grad_buffer)[2])
    else
        return Flux.Optimise.apply!(optim.optim, params, optim.grad_buffer)
    end
end

### generic FMIFlux.AbstractOptimiser ###


  