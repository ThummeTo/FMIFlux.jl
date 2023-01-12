#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Flux
import ChainRulesCore
import Flux.Random: AbstractRNG
import Flux.LinearAlgebra: I

# feed through
params = Flux.params

# exports
export Adam 
export Parallel

# 
Chain = Flux.Chain
export Chain

# Float64 version of the Flux.glorot_uniform
function glorot_uniform_64(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    scale = Float64(gain) * sqrt(24.0 / sum(Flux.nfan(dims...)))
    (rand(rng, Float64, dims...) .- 0.5) .* scale
end
glorot_uniform_64(dims::Integer...; kw...) = glorot_uniform_64(Flux.default_rng_value(), dims...; kw...)
glorot_uniform_64(rng::AbstractRNG=Flux.default_rng_value(); init_kwargs...) = (dims...; kwargs...) -> glorot_uniform_64(rng, dims...; init_kwargs..., kwargs...)  
ChainRulesCore.@non_differentiable glorot_uniform_64(::Any...)

# Float64 version of the Flux.identity_init
identity_init_64(cols::Integer; gain::Real=1, shift=0) = zeros(Float64, cols) # Assume bias
identity_init_64(rows::Integer, cols::Integer; gain::Real=1, shift=0) = circshift(Matrix{Float64}(I * gain, rows,cols), shift)
function identity_init_64(dims::Integer...; gain::Real=1, shift=0)
  nin, nout = dims[end-1], dims[end]
  centers = map(d -> cld(d, 2), dims[1:end-2])
  weights = zeros(Float64, dims...)
  for i in 1:min(nin,nout)
    weights[centers..., i, i] = gain
  end
  return circshift(weights, shift)
end
ChainRulesCore.@non_differentiable identity_init_64(::Any...)

"""
    Wrapper for Flux.Dense, that converts all parameters to Float64.
"""
function Dense(args...; init=glorot_uniform_64, kwargs...)
    return Flux.Dense(args...; init=init, kwargs...)
end
function Dense(W::AbstractMatrix, args...; init=glorot_uniform_64, kwargs...) 
    W = Matrix{Float64}(W)
    return Flux.Dense(W, args...; init=init, kwargs...)
end
Dense(in::Integer, out::Integer, σ = identity; init=glorot_uniform_64, kwargs...) = Dense(in => out, σ; init=init, kwargs...)
export Dense