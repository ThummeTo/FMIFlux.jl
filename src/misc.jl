#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

"""
Compares non-equidistant (or equidistant) datapoints by linear interpolating and comparing at given interpolation points `t_comp`. 
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
Writes/Copies flatted (Flux.destructure) training parameters `p_net` to non-flat model `net` with data offset `c`.
"""
function transferFlatParams!(net, p_net, c=1; netRange=nothing)
    
    if netRange == nothing
        netRange = 1:length(net.layers)
    end
    for l in netRange
        if !isa(net.layers[l], Flux.Dense)
            continue
        end
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

function transferParams!(net, p_net, c=1; netRange=nothing)
    
    if netRange == nothing
        netRange = 1:length(net.layers)
    end
    for l in netRange
        if !(net.layers[l] isa Flux.Dense)
            continue
        end

        for w in 1:length(net.layers[l].weight)
            net.layers[l].weight[w] = p_net[1+(l-1)*2][w]
        end
        
        for b in 1:length(net.layers[l].bias)
            net.layers[l].bias[b] = p_net[l*2][b]
        end
    end
end

# this is needed by Zygote, but not defined by default
function Base.ndims(::Tuple{Float64})
    return 1
end

# transposes a vector of vectors
function transpose(vec::AbstractVector{<:AbstractVector{<:Real}})
    return collect(eachrow(reduce(hcat, vec)))
end

function timeToIndex(ts::AbstractArray{<:Real}, target::Real)
    tStart = ts[1]
    tStop = ts[end]
    tLen = length(ts)

    @assert target >= tStart "timeToIndex(...): Time ($(target)) < tStart ($(tStart))"
    
    # because of the event handling condition, `target` can be outside of the simulation interval!
    # OLD: @assert target <= tStop "timeToIndex(...): Time ($(target)) > tStop ($(tStop))"
    # NEW:
    if target > tStop
        target = tStop
    end

    if target == tStart
        return 1
    elseif target == tStop
        return tLen
    end

    # estimate start value
    steps = 0
    i = min(max(round(Integer, (target-tStart)/(tStop-tStart)*tLen), 1), tLen)
    lastStep = Inf
    while !(ts[i] <= target && ts[i+1] > target)
        dt = target - ts[i] 
        step = round(Integer, dt/(tStop-tStart)*tLen)
        if abs(step) >= lastStep 
            step = Int(sign(dt))*(lastStep-1)
        end
        if step == 0
            step = Int(sign(dt))
        end
        lastStep = abs(step)

        #@info "$i  +=  $step  =  $(i+step)"
        i += step
        if i < 1
            i = 1
        elseif i > tLen
            i = tLen
        end
        steps += 1

        @assert steps < 200 "Steps reached max."
    end

    #@info "$steps"

    t = ts[i]
    next_t = ts[i+1]
    @assert t <= target && next_t >= target "No fitting time found, numerical issue."
    if (target-t) < (next_t-target)
        return i 
    else 
        return i+1
    end
    
end