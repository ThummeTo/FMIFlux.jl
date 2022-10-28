#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Statistics: mean, std

### SHIFTSCALE ###

struct ShiftScale{T}
    shift::AbstractArray{T}
    scale::AbstractArray{T}
    
    function ShiftScale{T}(shift::AbstractArray{T}, scale::AbstractArray{T}) where {T}
        inst = new(shift, scale)
        return inst
    end

    function ShiftScale(shift::AbstractArray{T}, scale::AbstractArray{T}) where {T}
        return ShiftScale{T}(shift, scale)
    end

    # initialize for data array
    function ShiftScale(data::AbstractArray{<:AbstractArray{T}}) where {T}
        shift = -mean.(data)
        scale = 1.0 ./ std.(data)
        return ShiftScale{T}(shift, scale)
    end
end
export ShiftScale

function (l::ShiftScale)(x)

    x_proc = (x .+ l.shift) .* l.scale
    
    return x_proc
end

Flux.@functor ShiftScale (shift, scale)

### SCALESHIFT ###

struct ScaleShift{T}
    scale::AbstractArray{T}
    shift::AbstractArray{T}
    
    function ScaleShift{T}(scale::AbstractArray{T}, shift::AbstractArray{T}) where {T}
        inst = new(scale, shift)
        return inst
    end

    function ScaleShift(scale::AbstractArray{T}, shift::AbstractArray{T}) where {T}
        return ScaleShift{T}(scale, shift)
    end

    # init ScaleShift with inverse transformation of a given ShiftScale
    function ScaleShift(l::ShiftScale{T}) where {T}
        return ScaleShift{T}(1.0 / l.scale, -1.0 * shift)
    end

    function ScaleShift(data::AbstractArray{<:AbstractArray{T}}) where {T}
        shift = mean.(data)
        scale = std.(data)
        return ShiftScale{T}(scale, shift)
    end
end
export ScaleShift

function (l::ScaleShift)(x)

    x_proc = (x .* l.scale) .+ l.shift
    
    return x_proc
end

Flux.@functor ScaleShift (scale, shift)

### CACHE ### 

mutable struct CacheLayer
    cache::AbstractArray

    function CacheLayer()
        inst = new()
        return inst
    end
end
export CacheLayer

function (l::CacheLayer)(x)

    l.cache = x
    
    return x
end

### CACHERetrieve ### 

struct CacheRetrieveLayer
    cacheLayer::CacheLayer
    
    function CacheRetrieveLayer(cacheLayer::CacheLayer)
        inst = new(cacheLayer)
        return inst
    end
end
export CacheRetrieveLayer

function (l::CacheRetrieveLayer)(idxBefore, x, idxAfter=[])
    return [l.cacheLayer.cache[idxBefore]..., x..., l.cacheLayer.cache[idxAfter]...]
end
