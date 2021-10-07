#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux

"""
Compares non-equidistant (or equdistant) datapoints by linear interpolating and comparing at given interpolation points `t_comp`. 
(Zygote-friendly: Zygote can differentiate through via AD.)
"""
function mse_interpolate(t1, x1, t2, x2, t_comp)
    #lin1 = LinearInterpolation(t1, x1)
    #lin2 = LinearInterpolation(t2, x2)
    ar1 = collect(lin_interp(t1, x1, t_sample) for t_sample in t_comp) #lin1.(t_comp)
    ar2 = collect(lin_interp(t2, x2, t_sample) for t_sample in t_comp) #lin2.(t_comp)
    Flux.Losses.mse(ar1, ar2)
end

# Helper: simple linear interpolation 
function lin_interp(t, x, t_sample)
    if t_sample <= t[1]
        return x[1]
    end

    if t_sample >= t[end]
        return x[end]
    end

    i = 1
    while t_sample > t[i]
        i += 1
    end

    x_left = x[i-1]
    x_right = x[i]

    t_left = t[i-1]
    t_right = t[i]

    dx = x_right - x_left
    dt = t_right - t_left
    h = t_sample - t_left

    x_left + dx/dt*h
end

"""
Writes/Copies training parameters from `p_net` to `net` with data offset `c`.
"""
function transferParams!(net, p_net, c=0)
    numLayers = length(net.layers)
    for l in 1:numLayers
        ni = size(net.layers[l].weight,2)
        no = size(net.layers[l].weight,1)

        w = zeros(no, ni)
        b = zeros(no)

        for i in 1:ni
            for o in 1:no
                w[o,i] = p_net[1][c + (i-1)*no + (o-1)]
            end
        end

        c += ni*no

        for o in 1:no
            b[o] = p_net[1][c + (o-1)]
        end

        c += no

        copy!(net.layers[l].weight, w)
        copy!(net.layers[l].bias, b)
    end
end
