#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Flux

function need_convert(::Type{T}, obj::Union{AbstractMatrix{O}, AbstractVector{O}}) where {T, O}
    if T == O
        return false
    else
        return true
    end
end

function convert(::Type{T}, obj::Union{AbstractMatrix{O}, AbstractVector{O}}) where {T, O}
    return typeof(obj).name.wrapper{T}(obj)
end

function convert(::Type{T}, layers::Union{AbstractArray, Tuple}; additionalFields::Tuple=()) where {T}

    convert_fields = (:weight, :bias, :scale, :shift, additionalFields...)
    newlayers = []

    for layer in layers
        typ = typeof(layer)
        fields = []

        # for all fields of struct
        for fieldname in fieldnames(typ)
            
            value = getfield(layer, fieldname)

            if fieldname ∈ convert_fields
                if need_convert(T, value)
                    value = convert(T, value)
                end
            end

            push!(fields, value)
        end

        newlayer = typ.name.wrapper(fields...)
        push!(newlayers, newlayer)
        logInfo(fmu, "Succesfully converted layer of type `$typ` to `$(typeof(newlayer))`.")
    end

    logInfo(fmu, "ME_NeuralFMU(...): Succesfully converted model to `$(T)`.")

    if isa(layers, Tuple)
        return (newlayers...,)
    else
        return newlayers
    end
end 
export convert

function convert(::Type{T}, chain::Flux.Chain) where {T}

    newlayers = convert(T, chain.layers)
    return Flux.Chain(newlayers...)
end 

function need_convert(::Type{T}, layers::Union{AbstractArray, Tuple}) where {T}

    convert_fields = (:weight, :bias, :scale, :shift)
   
    for layer in layers
        typ = typeof(layer)

        # for all fields of struct
        for fieldname in fieldnames(typ)
            
            value = getfield(layer, fieldname)

            if fieldname ∈ convert_fields
                if need_convert(T, value)
                    return true 
                end
            end

        end
    end

    return false
end 
