#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module OptimExt

using FMIFlux
import Optim

import FMIFlux: AbstractOptimiser

### Optim.jl ###

struct OptimOptimiserWrapper{G} <: AbstractOptimiser
    optim::Optim.AbstractOptimizer
    grad_fun!::G

    state::Union{Optim.AbstractOptimizerState,Nothing}
    d::Union{Optim.OnceDifferentiable,Nothing}
    options::Any

    function FMIFlux.OptimOptimiserWrapper(
        optim::Optim.AbstractOptimizer,
        grad_fun!::G,
        loss,
        params,
    ) where {G}
        options = Optim.Options(
            outer_iterations = 1,
            iterations = 1,
            g_calls_limit = 1,
            f_calls_limit = 5,
        )

        # should be ignored anyway, because function `g!` is given
        autodiff = :forward # = ::finite
        inplace = true

        d = Optim.promote_objtype(optim, params, autodiff, inplace, loss, grad_fun!)
        state = Optim.initial_state(optim, options, d, params)

        return new{G}(optim, grad_fun!, state, d, options)
    end

end
export OptimOptimiserWrapper

function FMIFlux.apply!(optim::OptimOptimiserWrapper, params)

    res = Optim.optimize(optim.d, params, optim.optim, optim.options, optim.state)
    step = params .- Optim.minimizer(res)

    return step
end

# Dispatch for Optim.jl [Optim.AbstractOptimizer]
function FMIFlux._train!(
    loss,
    params, #::Union{Flux.Params,Zygote.Params,AbstractVector{<:AbstractVector{<:Real}}},
    data,
    optim::Optim.AbstractOptimizer;
    gradient::Symbol = :ReverseDiff,
    chunk_size::Union{Integer,Symbol} = :auto_fmiflux,
    multiObjective::Bool = false,
    kwargs...,
)
    if length(params) <= 0 # || length(params[1]) <= 0
        @warn "train!(...): Empty parameter array, training on an empty parameter array doesn't make sense."
        return
    end

    grad_fun! =
        (G, p) -> FMIFlux.computeGradient!(G, loss, p, gradient, chunk_size, multiObjective)
    _optim = FMIFlux.OptimOptimiserWrapper(optim, grad_fun!, loss, params)
    FMIFlux._train!(
        loss,
        params,
        data,
        _optim;
        gradient = gradient,
        chunk_size = chunk_size,
        multiObjective = multiObjective,
        kwargs...,
    )
end

end # module OptimExt
