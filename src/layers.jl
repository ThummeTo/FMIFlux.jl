#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Statistics: mean, std

### FMUParameterRegistrator ###

"""
ToDo.
"""
struct FMUParameterRegistrator{T}
    fmu::FMU2
    p_refs::AbstractArray{<:fmi2ValueReference}
    p::AbstractArray{T}
   
    function FMUParameterRegistrator{T}(fmu::FMU2, p_refs::fmi2ValueReferenceFormat, p::AbstractArray{T}) where {T}
        @assert length(p_refs) == length(p) "`p_refs` and `p` need to be the same length!"
        p_refs = prepareValueReference(fmu, p_refs)

        fmu.default_p_refs = p_refs 
        fmu.default_p = p 
        for c in fmu.components
            c.default_p_refs = p_refs
            c.default_p = p
        end

        return new(fmu, p_refs, p)
    end

    function FMUParameterRegistrator(fmu::FMU2, p_refs::fmi2ValueReferenceFormat, p::AbstractArray{T}) where {T}
        return FMUParameterRegistrator{T}(fmu, p_refs, p)
    end
end
export FMUParameterRegistrator

function (l::FMUParameterRegistrator)(x)
    
    l.fmu.default_p_refs = l.p_refs
    l.fmu.default_p = l.p 
    for c in l.fmu.components
        c.default_p_refs = l.p_refs
        c.default_p = l.p
    end
    
    return x
end

Flux.@functor FMUParameterRegistrator (p, )

### SHIFTSCALE ###

"""
ToDo.
"""
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
    function ShiftScale(data::AbstractArray{<:AbstractArray{T}}; range::Union{Symbol, UnitRange{<:Integer}}=-1:1) where {T}
        shift = -mean.(data)
        scale = nothing

        if range == :NormalDistribution
            scale = 1.0 ./ std.(data)
        elseif isa(range, UnitRange{<:Integer})
            scale = 1.0 ./ (collect(max(d...) for d in data) - collect(min(d...) for d in data)) .* (range[end] - range[1])
        else
            @assert false "Unsupported scaleMode, supported is `:NormalDistribution` or `UnitRange{<:Integer}`"
        end

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

"""
ToDo.
"""
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
    function ScaleShift(l::ShiftScale{T}; indices=1:length(l.scale)) where {T}
        return ScaleShift{T}(1.0 ./ l.scale[indices], -1.0 .* l.shift[indices])
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

### ScaleSum ###

struct ScaleSum{T}
    scale::AbstractArray{T}
    groups::Union{AbstractVector{<:AbstractVector{<:Integer}}, Nothing}
    
    function ScaleSum{T}(scale::AbstractArray{T}, groups::Union{AbstractVector{<:AbstractVector{<:Integer}}, Nothing}=nothing) where {T}
        inst = new(scale, groups)
        return inst
    end

    function ScaleSum(scale::AbstractArray{T}, groups::Union{AbstractVector{<:AbstractVector{<:Integer}}, Nothing}=nothing) where {T}
        return ScaleSum{T}(scale, groups)
    end
end
export ScaleSum

function (l::ScaleSum)(x)

    if isnothing(l.groups)
        x_proc = sum(x .* l.scale)
        return [x_proc]
    else
        return collect(sum(x[g] .* l.scale[g]) for g in l.groups)
    end
end

Flux.@functor ScaleSum (scale, )

### CACHE ### 

mutable struct CacheLayer
    cache::AbstractArray{<:AbstractArray}

    function CacheLayer()
        inst = new()
        inst.cache = Array{Array,1}(undef, Threads.nthreads())
        return inst
    end
end
export CacheLayer

function (l::CacheLayer)(x)

    tid = Threads.threadid()
    l.cache[tid] = x
    
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

function (l::CacheRetrieveLayer)(idxBefore, x=nothing, idxAfter=nothing)
    tid = Threads.threadid()

    # Zygote doesn't like empty arrays
    if idxAfter == nothing && x == nothing
        return l.cacheLayer.cache[tid][idxBefore]
    elseif idxAfter == nothing
        return [l.cacheLayer.cache[tid][idxBefore]..., x...]
    elseif x == nothing
        return [l.cacheLayer.cache[tid][idxBefore]..., l.cacheLayer.cache[tid][idxAfter]...]
    else
        return [l.cacheLayer.cache[tid][idxBefore]..., x..., l.cacheLayer.cache[tid][idxAfter]...]
    end
end
