#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

"""
Plots a NeuralFMU (ME).
"""
function fmiPlot(nfmu::NeuralFMU; kwargs...)
    FMI.fmiPlot(nfmu.solution; kwargs...)
end

function fmiPlot!(fig, nfmu::NeuralFMU; kwargs...)
    FMI.fmiPlot!(nfmu.solution; kwargs...)
end

FMI.fmiPlot(nfmu::NeuralFMU) = fmiPlot(nfmu)
