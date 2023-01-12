#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Flux.Optimisers

struct SoftStart{T} <: Optimisers.AbstractRule
    minx::T 
    maxx::T
    steps::UInt

    function SoftStart{T}(minx::T, steps::UInt; maxx::T=1.0) where {T}
        inst = new()
        inst.minx = minx
        inst.maxx = maxx
        inst.steps = steps
        return inst
    end

    function SoftStart(minx::T, steps::UInt; maxx::T=1.0)
        return SoftStart{fmi2Real}(minx, steps; maxx=maxx)
    end
end
export SoftStart
  
function Optimisers.apply!(o::SoftStart, state, x, x̄)
    step = state

    if step > o.steps
        newx̄ = o.maxx
    else
        newx̄ = o.minx * ((o.maxx/o.minx)^(1.0/o.steps*step))
    end

    nextstate = step + 1
    return nextstate, newx̄
end
  
function Optimisers.init(o::SoftStart, x::AbstractArray)
    return 0
end