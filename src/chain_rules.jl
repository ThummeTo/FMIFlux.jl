#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMISensitivity.FMICore.ChainRulesCore
import FMISensitivity.ForwardDiffChainRules: @ForwardDiff_frule
import FMISensitivity.SciMLSensitivity.ReverseDiff: @grad_from_chainrules

function ChainRulesCore.frule(Δtuple, 
    ::typeof(stateChangeByEvent!), 
    xBuf, 
    nfmu, 
    c, 
    left_x)

    Δself, ΔxBuf, Δnfmu, Δc, Δleft_x = undual(Δtuple)

    Ω = stateChangeByEvent!(xBuf, nfmu, c, left_x)

    ∂Ω = Ω

    return Ω, ∂Ω 
end

function ChainRulesCore.rrule(::typeof(stateChangeByEvent!), 
    xBuf, 
    nfmu, 
    c, 
    left_x)

    Ω = stateChangeByEvent!(xBuf, nfmu, c, left_x)

    ##############

    function eval_pullback(r̄)
 
        x̄Buf = r̄

        # write back
        f̄ = NoTangent()
        x̄Buf = r̄
        n̄fmu = ZeroTangent()
        c̄ = NoTangent()
        l̄eft_x = Ω * r̄
        
        # [ToDo] This needs to be a tuple... but this prevents pre-allocation...
        return (f̄, x̄Buf, n̄fmu, c̄, l̄eft_x)
    end

    return (Ω, eval_pullback)
end

@ForwardDiff_frule eval!(xBuf::AbstractVector{<:ForwardDiff.Dual},
    nfmu, 
    c, 
    left_x)

@grad_from_chainrules eval!(xBuf::AbstractVector{<:ReverseDiff.TrackedReal},
    nfmu, 
    c, 
    left_x)