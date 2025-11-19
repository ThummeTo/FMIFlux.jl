#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

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

    # i = 1 
    # while ts[i] < target
    #     i += 1
    # end 
    # return i

    # estimate start value
    steps = 0
    i = min(max(round(Integer, (target - tStart) / (tStop - tStart) * tLen), 1), tLen)
    lastStep = Inf
    while !(ts[i] <= target && ts[i+1] > target)
        dt = target - ts[i]
        step = round(Integer, dt / (tStop - tStart) * tLen)
        if abs(step) >= lastStep
            step = Int(sign(dt)) * (lastStep - 1)
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

        @assert steps < tLen "Steps reached max."
    end

    #@info "$steps"

    t = ts[i]
    next_t = ts[i+1]
    @assert t <= target && next_t >= target "No fitting time found, numerical issue."
    if (target - t) < (next_t - target)
        return i
    else
        return i + 1
    end

end
