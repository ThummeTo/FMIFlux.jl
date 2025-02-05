#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module JLD2Ext

using FMIFlux, JLD2

"""
    Saves parameters for a neural FMU.
"""
function FMIFlux.saveParameters(nfmu::NeuralFMU, path::String; keyword = "parameters")

    params = FMIFlux.params(nfmu)
    JLD2.save(path, Dict(keyword => params))
end

"""
    Loads parameters for a neural FMU.
"""
function FMIFlux.loadParameters(
    nfmu::NeuralFMU,
    path::String;
    keyword = "parameters",
)
    nfmu.p = JLD2.load(path, keyword)
    return nothing
end

end # JLD2Ext
