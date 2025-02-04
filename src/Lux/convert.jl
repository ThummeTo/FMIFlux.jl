#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# in Lux.jl, only operations are defined, and every model is Float32/64 at the same time
function FMIFlux.is64(model::Lux.Chain)
    return true
end

# in Lux.jl, only operations are defined, and every model is Float32/64 at the same time
function FMIFlux.convert64(model::Lux.Chain)
    return model 
end

function FMIFlux.destructure(model::Lux.Chain)
    rng = Random.default_rng()
    p, st = Lux.setup(rng, model)
    return p, st
end

function FMIFlux.eval(nfmu::ME_NeuralFMU{M,R}, input; p=nfmu.p) where {M <: Lux.Chain, R}
    first(nfmu.model(input, p, nfmu.re))
end