#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Plots

function fmiPlot(nfmu::NeuralFMU)
    FMI.fmiPlot(nfmu.fmu, nfmu.solution, true)
end

FMI.fmiPlot(nfmu::NeuralFMU) = fmiPlot(nfmu)
Plots.plot(nfmu::NeuralFMU) = fmiPlot(nfmu)
