#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module JLD2Ext

using FMIFlux, JLD2

function FMIFlux.saveParameters(nfmu::NeuralFMU, path::String; keyword = "parameters")

    params = FMIFlux.params(nfmu)

    JLD2.save(path, Dict(keyword => params[1]))
end

function FMIFlux.loadParameters(
    nfmu::NeuralFMU,
    path::String;
    flux_model = nothing,
    keyword = "parameters",
)

    paramsLoad = JLD2.load(path, keyword)

    nfmu_params = FMIFlux.params(nfmu)
    flux_model_params = nothing

    if flux_model != nothing
        flux_model_params = Flux.params(flux_model)
    end

    numParams = length(nfmu_params)
    l = 1
    p = 1
    for i = 1:numParams
        nfmu_params[i] = paramsLoad[i]

        if flux_model != nothing
            flux_model_params[l][p] = paramsLoad[i]

            p += 1

            if p > length(flux_model_params[l])
                l += 1
                p = 1
            end
        end
    end

    return nothing
end

end # JLD2Ext
