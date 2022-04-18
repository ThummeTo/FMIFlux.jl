#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMIImport: fmi2Real

import FMIImport: fmi2SetTime, fmi2SetContinuousStates, fmi2SetReal
import FMIImport: fmi2CompletedIntegratorStep, fmi2NewDiscreteStates!, fmi2EventInfo, fmi2GetEventIndicators!
import FMIImport: fmi2GetReal, fmi2GetDerivatives, fmi2GetContinuousStates
import FMIImport: fmi2EnterContinuousTimeMode, fmi2EnterEventMode

using FMIImport: fmi2ComponentStateContinuousTimeMode, fmi2ComponentStateEventMode

mutable struct FMU2ComponentShadow 

    # compatibility 
    t::Float64 
    realComponent
    jacobianUpdate!
    compAddr
    log::Bool

    eventInfo::fmi2EventInfo
    state
    fmu

    jac_ẋy_x::Matrix{fmi2Real}
    jac_ẋy_u::Matrix{fmi2Real}
    jac_x::Array{fmi2Real}
    jac_u::Union{Array{fmi2Real}, Nothing}
    jac_t::fmi2Real

    # derivative vector caching
    t_ẋ_cache::Array{Float64, 1}
    ẋ_cache::Array{Array{Float64, 1}, 1}
    ẋ_interp  

    t_y_cache::Array{Float64, 1}
    y_cache::Array{Array{Float64, 1}, 1}
    y_interp   
    
    t_x_cache::Array{Float64, 1}
    x_cache::Array{Array{Float64, 1}, 1}
    x_interp   

    t_jac_ẋy_x_cache::Array{Float64, 1}
    jac_ẋy_x_cache::Array{Matrix{Float64}, 1}
    jac_ẋy_x_interp

    t_jac_ẋy_u_cache::Array{Float64, 1}
    jac_ẋy_u_cache::Array{Matrix{Float64}, 1}
    jac_ẋy_u_interp

    function FMU2ComponentShadow() 
        inst = new()
        inst.log = false
        inst.realComponent = nothing

        inst.compAddr = C_NULL

        inst.jacobianUpdate! = jacobianUpdate!

        inst.jac_x = Array{fmi2Real, 1}()
        inst.jac_u = nothing
        inst.jac_t = -1.0
        inst.jac_ẋy_x = zeros(fmi2Real, 0, 0)
        inst.jac_ẋy_u = zeros(fmi2Real, 0, 0)
        
        reset!(inst)

        return inst 
    end
end

function setComponent!(comp::FMU2ComponentShadow, realComponent::Union{FMU2Component, Nothing})

    comp.realComponent = realComponent

    if realComponent != nothing
        comp.eventInfo = realComponent.eventInfo
        comp.state = realComponent.state
        comp.fmu = realComponent.fmu
        comp.t = realComponent.t
        comp.compAddr = realComponent.compAddr
    end
end

function jacobianUpdate!(mat, comp::FMU2ComponentShadow, rdx, rx)
   
    if rx == comp.fmu.modelDescription.stateValueReferences

        if comp.jac_ẋy_x_interp !== nothing
            @assert length(comp.jac_ẋy_x_interp) > 0 "empty interpolation polynominal"

            b = bounds(comp.jac_ẋy_x_interp.itp)[1]
            t = comp.t 
            t = max(b[1], min(b[2], t))
            mat[:] = comp.jac_ẋy_x_interp(t)
            
        else
            comp.realComponent.jacobianUpdate!(mat, comp.realComponent, rdx, rx)

            if isLogging(comp)
                if length(comp.t_jac_ẋy_x_cache) == 0 || comp.t_jac_ẋy_x_cache[end] < comp.t
                    push!(comp.t_jac_ẋy_x_cache, comp.t)
                    push!(comp.jac_ẋy_x_cache, copy(mat))
                end
            end
        end

    elseif rx == comp.fmu.modelDescription.inputValueReferences

        if comp.jac_ẋy_u_interp !== nothing
            @assert length(comp.jac_ẋy_u_interp) > 0 "empty interpolation polynominal"
            
            b = bounds(comp.jac_ẋy_u_interp.itp)[1]
            t = comp.t 
            t = max(b[1], min(b[2], t))
            mat[:] = comp.jac_ẋy_u_interp(t) 
        
        else
            comp.realComponent.jacobianUpdate!(mat, comp.realComponent, rdx, rx)

            if isLogging(comp)
                if length(comp.t_jac_ẋy_u_cache) == 0 || comp.t_jac_ẋy_u_cache[end] < comp.t
                    push!(comp.t_jac_ẋy_u_cache, comp.t)
                    push!(comp.jac_ẋy_u_cache, copy(mat))
                end
            end
        end

    else
        @assert false "shadow jacobianUpdate!, unknown rx = $(rx)"
    end

end

function reset!(componentShadow::FMU2ComponentShadow)
    componentShadow.t_y_cache = [] 
    componentShadow.t_x_cache = []
    componentShadow.t_ẋ_cache = [] 
    componentShadow.t_jac_ẋy_x_cache = [] 
    componentShadow.t_jac_ẋy_u_cache = [] 

    componentShadow.y_cache = []
    componentShadow.x_cache = []
    componentShadow.ẋ_cache = []
    componentShadow.jac_ẋy_x_cache = []
    componentShadow.jac_ẋy_u_cache = []

    componentShadow.y_interp = nothing
    componentShadow.x_interp = nothing
    componentShadow.ẋ_interp = nothing
    componentShadow.jac_ẋy_x_interp = nothing
    componentShadow.jac_ẋy_u_interp = nothing
end

function prepare!(componentShadow::FMU2ComponentShadow)

    # if length(componentShadow.ẋ_cache) == 0 || length(componentShadow.jac_ẋy_x_cache) == 0
    #     return nothing
    # end

    if length(componentShadow.ẋ_cache) > 0 
        componentShadow.ẋ_interp = LinearInterpolation(componentShadow.t_ẋ_cache, componentShadow.ẋ_cache)
        @debug "[ẋ_interp]"
    end

    if length(componentShadow.x_cache) > 0 
        componentShadow.x_interp = LinearInterpolation(componentShadow.t_x_cache, componentShadow.x_cache)
        @debug "[x_interp]"
    end

    if length(componentShadow.y_cache) > 0 
        componentShadow.y_interp = LinearInterpolation(componentShadow.t_y_cache, componentShadow.y_cache)
        @debug "[y_interp]"
    end

    if length(componentShadow.jac_ẋy_x_cache) > 0
        componentShadow.jac_ẋy_x_interp = LinearInterpolation(componentShadow.t_jac_ẋy_x_cache, componentShadow.jac_ẋy_x_cache)
        @debug "[jac_ẋy_x_interp]"
    end

    if length(componentShadow.jac_ẋy_u_cache) > 0 
        componentShadow.jac_ẋy_u_interp = LinearInterpolation(componentShadow.t_jac_ẋy_u_cache, componentShadow.jac_ẋy_u_cache) 
        @debug "[jac_ẋy_u_interp]"
    end

    # if componentShadow.realComponent != nothing 
    #     if componentShadow.fmu.executionConfig.freeInstance
    #         fmi2FreeInstance!(componentShadow.realComponent)
    #     end 
    #     componentShadow.realComponent = nothing
    # end
end

function isPrepared(componentShadow::FMU2ComponentShadow)
    return (componentShadow.ẋ_interp !== nothing)
end

function isLogging(componentShadow::FMU2ComponentShadow)
    return componentShadow.log
end

function fmi2SetTime(comp::FMU2ComponentShadow, t)
    # if isPrepared(comp)
    #     comp.t = t
    #     return fmi2StatusOK
    # else
        comp.t = t 
        status = fmi2SetTime(comp.realComponent, t)
        comp.state = comp.realComponent.state
        return status
    # end
end

function fmi2EnterContinuousTimeMode(comp::FMU2ComponentShadow)
    # if isPrepared(comp)
    #     comp.state = fmi2ComponentStateContinuousTimeMode
    #     return fmi2StatusOK
    # else 
        status = fmi2EnterContinuousTimeMode(comp.realComponent)
        comp.state = comp.realComponent.state
        return status
    # end 
end

function fmi2EnterEventMode(comp::FMU2ComponentShadow)
    # if isPrepared(comp)
    #     comp.state = fmi2ComponentStateEventMode
    #     return fmi2StatusOK
    # else 
        status = fmi2EnterEventMode(comp.realComponent)
        comp.state = comp.realComponent.state
        return status
    # end 
end

function fmi2SetContinuousStates(comp::FMU2ComponentShadow, x)
    # if isPrepared(comp)
    #     # do nothing 
    #     return fmi2StatusOK
    # else 
        return fmi2SetContinuousStates(comp.realComponent, x)
    # end 
end

function fmi2SetReal(comp::FMU2ComponentShadow, vrs, vals)
    # if isPrepared(comp)
    #     # do nothing 
    #     return fmi2StatusOK
    # else
        return fmi2SetReal(comp.realComponent, vrs, vals)
    # end
end

function fmi2CompletedIntegratorStep(comp::FMU2ComponentShadow, val)
    # if isPrepared(comp)
    #     status = fmi2StatusOK
    #     enterEventMode = fmi2False 
    #     terminateSimulation = fmi2False
    #     return (status, enterEventMode, terminateSimulation)
    # else
        status = fmi2CompletedIntegratorStep(comp.realComponent, val)
        comp.state = comp.realComponent.state
        return status
    #end
end

function fmi2NewDiscreteStates!(comp::FMU2ComponentShadow, eventInfo)

    # if isPrepared(comp)
    #     eventInfo.newDiscreteStatesNeeded = fmi2False
    #     eventInfo.terminateSimulation = fmi2False
    #     eventInfo.nominalsOfContinuousStatesChanged = fmi2False
    #     eventInfo.valuesOfContinuousStatesChanged = fmi2False
    #     eventInfo.nextEventTimeDefined = fmi2False
    #     eventInfo.nextEventTime = 0.0

    #     return fmi2StatusOK
    # else 
        status = fmi2NewDiscreteStates!(comp.realComponent, eventInfo)
        comp.state = comp.realComponent.state
        return status
    #end
end

function fmi2GetEventIndicators!(comp::FMU2ComponentShadow, out)
    # if isPrepared(comp)
    #     for i in 1:length(out)
    #         out[i] = 1.0 
    #     end
    #     return fmi2StatusOK
    # else
        return fmi2GetEventIndicators!(comp.realComponent, out)
    #end
end

function fmi2GetReal(comp::FMU2ComponentShadow, getValueReferences)
    
    y = nothing 

    if (comp.y_interp !== nothing)

        b = bounds(comp.y_interp.itp)[1]
        t = comp.t 
        t = max(b[1], min(b[2], t))
        y = comp.y_interp(t)
        
    else

        y = fmi2GetReal(comp.realComponent, getValueReferences)

        if isLogging(comp)
            if length(comp.t_y_cache) == 0 || comp.t_y_cache[end] < comp.t
                push!(comp.y_cache, collect(ForwardDiff.value(e) for e in y) )
                push!(comp.t_y_cache, comp.t)
            end
        end
    end

    return y
end

function fmi2GetDerivatives(comp::FMU2ComponentShadow)
    
    ẋ = nothing 

    if (comp.ẋ_interp !== nothing)

        b = bounds(comp.ẋ_interp.itp)[1]
        t = comp.t 
        t = max(b[1], min(b[2], t))
        ẋ = comp.ẋ_interp(t)
       
    else

        ẋ = fmi2GetDerivatives(comp.realComponent)

        if isLogging(comp)
            if length(comp.t_ẋ_cache) == 0 || comp.t_ẋ_cache[end] < comp.t
                push!(comp.ẋ_cache, collect(ForwardDiff.value(e) for e in ẋ) )
                push!(comp.t_ẋ_cache, comp.t)
            end
        end

    end

    #@info "derivatives, prepared=$(isPrepared(comp)), ẋ=$(ẋ), length=$(length(comp.t_ẋ_cache))"

    return ẋ
end

function fmi2GetContinuousStates(comp::FMU2ComponentShadow)
    
    x = nothing 

    if (comp.x_interp !== nothing)

        b = bounds(comp.x_interp.itp)[1]
        t = comp.t 
        t = max(b[1], min(b[2], t))
        x = comp.x_interp(t)

    else

        x = fmi2GetContinuousStates(comp.realComponent)

        if isLogging(comp)
            if length(comp.t_x_cache) == 0 || comp.t_x_cache[end] < comp.t
                push!(comp.x_cache, x)
                push!(comp.t_x_cache, comp.t)
            end
        end

    end

    return x
end