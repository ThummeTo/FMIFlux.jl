#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Plots

function fmiPlot(nfmu::ME_NeuralFMU)
    FMI.fmiPlot(nfmu.fmu, nfmu.solution, true)
end

function fmiPlot(nfmu::CS_NeuralFMU)
    FMI.fmiPlot(nfmu.simulationResult)
end

FMI.fmiPlot(nfmu::NeuralFMUs) = fmiPlot(nfmu)
Plots.plot(nfmu::NeuralFMUs) = fmiPlot(nfmu)
