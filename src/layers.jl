#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

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
end
export ShiftScale

function (l::ShiftScale)(x)

    x_proc = (x .+ m.shift) .* m.scale
    
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
end
export ScaleShift

function (l::ScaleShift)(x)

    x_proc = (x .* m.scale) .+ m.shift
    
    return x_proc
end

Flux.@functor ScaleShift (scale, shift)

### SKIPSTART ### 

struct SkipStart{T}
    cache::AbstractArray{T}
    indices::AbstractArray{UInt}
    map::Dict{UInt, UInt}
    
    function SkipStart{T}(cache::AbstractArray{T}, indices::AbstractArray{UInt}) where {T}
        map = Dict()
        mi = 1
        for i in 1:max(indices)
            if i ∈ indices
                map[i] = mi
                mi += 1
            end
        end
        inst = new(cache, indices, map)
        return inst
    end

    function ScaleShift(cache::AbstractArray{T}, indices::AbstractArray{UInt}) where {T}
        return ScaleShift{T}(cache, indices)
    end
end
export SkipStart

function (l::SkipStart)(x)

    x.cache = x
    
    return x[m.indices]
end

### SKIPSTOP ### 

struct SkipStop{T}
    start::SkipStart{T}
    
    function SkipStop{T}(start::SkipStart{T}) where {T}
        inst = new(start)
        return inst
    end

    function SkipStop(start::SkipStart{T}) where {T}
        return SkipStop{T}(start)
    end
end
export SkipStop

function (l::SkipStop)(x)
    
    return collect((i ∈ l.start.indices ? x[l.start.map[i]] : l.start.cache[i]) for i in 1:length(l.start.cache))
end
