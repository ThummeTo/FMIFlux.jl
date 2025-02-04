#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

"""
Writes/Copies flatted (Flux.destructure) training parameters `p_net` to non-flat model `net` with data offset `c`.
"""
function transferFlatParams!(net, p_net, c = 1; netRange = nothing)

    if netRange == nothing
        netRange = 1:length(net.layers)
    end
    for l in netRange
        if !isa(net.layers[l], Flux.Dense)
            continue
        end
        ni = size(net.layers[l].weight, 2)
        no = size(net.layers[l].weight, 1)

        w = zeros(no, ni)
        b = zeros(no)

        for i = 1:ni
            for o = 1:no
                w[o, i] = p_net[1][c+(i-1)*no+(o-1)]
            end
        end

        c += ni * no

        for o = 1:no
            b[o] = p_net[1][c+(o-1)]
        end

        c += no

        copy!(net.layers[l].weight, w)
        copy!(net.layers[l].bias, b)
    end
end

function transferParams!(net, p_net, c = 1; netRange = nothing)

    if netRange == nothing
        netRange = 1:length(net.layers)
    end
    for l in netRange
        if !(net.layers[l] isa Flux.Dense)
            continue
        end

        for w = 1:length(net.layers[l].weight)
            net.layers[l].weight[w] = p_net[1+(l-1)*2][w]
        end

        for b = 1:length(net.layers[l].bias)
            net.layers[l].bias[b] = p_net[l*2][b]
        end
    end
end

# # this is needed by Zygote, but not defined by default
# function Base.ndims(::Tuple{Float64})
#     return 1
# end

# # transposes a vector of vectors
# function transpose(vec::AbstractVector{<:AbstractVector{<:Real}})
#     return collect(eachrow(reduce(hcat, vec)))
# end


