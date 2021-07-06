#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Plots

"""
Plots a NeuralFMU (ME).
"""
function fmiPlot(nfmu::ME_NeuralFMU)
    FMI.fmiPlot(nfmu.fmu, nfmu.solution, true)
end

"""
Plots a NeuralFMU (CS).
"""
function fmiPlot(nfmu::CS_NeuralFMU)
    numSig = length(nfmu.valueStack[1])-1
    t = collect(data[1] for data in nfmu.valueStack)

    fig = Plots.plot(xlabel="t [s]")
    for i in 1:numSig 
        values = collect(data[i+1] for data in nfmu.valueStack)
        Plots.plot!(fig, t, values, label="NeuralFMU output #$(i)")
    end 

    fig
end

# extends the existing plot-commands.
FMI.fmiPlot(nfmu::NeuralFMUs) = fmiPlot(nfmu)
Plots.plot(nfmu::NeuralFMUs) = fmiPlot(nfmu)
