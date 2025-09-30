#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

function FMIFlux.is64(model::Flux.Chain)
   
    for l = 1:length(model) # layers 
        #@info "Layer $(l)" # \n$(model[l])
        params, re = Flux.destructure(model[l])

        for i in 1:length(params)
            if !isa(params[i], Float64)
                return false
            end
        end
    end

    return true
end

function FMIFlux.convert64(model::Flux.Chain)
    Flux.fmap(Flux.f64, model)
end

function FMIFlux.destructure(model::Flux.Chain)
    Flux.destructure(model)
end

# adapting the Flux functions
function FMIFlux.params(nfmu::ME_NeuralFMU; destructure::Bool = false)
    
    #### DEPRECATED 

    if destructure || isnothing(nfmu.p)
        nfmu.p, nfmu.re = Flux.destructure(nfmu.model)
    end

    # ps = Flux.params(nfmu.p)

    # if issense(ps)
    #     @warn "Parameters include AD-primitives, this indicates that something did go wrong in before."
    # end

    # return ps

    return nfmu.p
end

function FMIFlux.params(nfmu::CS_NeuralFMU; destructure::Bool = false) # true)
   
    #### DEPRECATED 

    if destructure || isnothing(nfmu.p)
        nfmu.p, nfmu.re = Flux.destructure(nfmu.model)

        # else
        #     return Flux.params(nfmu.model)
    end

    # ps = Flux.params(nfmu.p)

    # if issense(ps)
    #     @warn "Parameters include AD-primitives, this indicates that something did go wrong in before."
    # end

    # return ps

    return nfmu.p
end

function FMIFlux.eval(nfmu::ME_NeuralFMU{M, R}, input; p=nfmu.p) where {M <: Flux.Chain, R}
    return nfmu.re(p)(input)
end

function FMIFlux.eval(nfmu::CS_NeuralFMU{F, C}, input; p=nfmu.p) where {F, C} # {F <:FMU2, C <:FMU2Component}
    return nfmu.re(p)(input)
end

# function FMIFlux.eval(nfmu::CS_NeuralFMU{F, C}, input; p=nfmu.p) where {F <: Vector{<:FMU2}, C <: Vector{<:FMU2Component}}
#     return nfmu.re(p)(input)
# end
