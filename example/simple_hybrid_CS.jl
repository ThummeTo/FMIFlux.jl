#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

################################## INSTALLATION ####################################################
# (1) Enter Package Manager via     ]
# (2) Install FMI via               add FMI       or   add "https://github.com/ThummeTo/FMI.jl"
# (3) Install FMIFlux via           add FMIFlux   or   add "https://github.com/ThummeTo/FMIFlux.jl"
################################ END INSTALLATION ##################################################

# this example covers creation and training am CS-NeuralFMUs

using FMI
using FMIFlux
using Flux
using DifferentialEquations: Tsit5
import Plots

FMUPath = joinpath(dirname(@__FILE__), "../model/SpringPendulumExtForce1D.fmu")

t_start = 0.0
t_step = 0.01
t_stop = 5.0

myFMU = fmiLoad(FMUPath)
fmiInstantiate!(myFMU; loggingOn=false)

fmiSetupExperiment(myFMU, t_start, t_stop)
fmiSetReal(myFMU, "mass_s0", 1.3)   # increase amplitude, invert phase
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

realSimData = fmi2SimulateCS(myFMU, t_step, t_start, t_stop, ["mass.s", "mass.v", "mass.a"], false)
fmiPlot(realSimData)

fmiReset(myFMU)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

fmuSimData = fmi2SimulateCS(myFMU, t_step, t_start, t_stop, ["mass.s", "mass.v", "mass.a"], false)
fmiPlot(fmuSimData)

######

tData = t_start:t_step:t_stop
extF = zeros(length(tData)) # no external force
posData = fmi2SimulationResultGetValues(realSimData, "mass.s")
velData = fmi2SimulationResultGetValues(realSimData, "mass.v")
accData = fmi2SimulationResultGetValues(realSimData, "mass.a")

# loss function for training
function losssum()
    solution = problem(t_start, t_step, t_stop, extF)

    accNet = collect(data[2] for data in solution)
    #velNet = collect(data[3] for data in solution)

    Flux.Losses.mse(accNet, accData) #+ Flux.Losses.mse(velNet, velData)
end

# callback function for training
global iterCB = 0
function callb()
    global iterCB += 1

    if iterCB % 10 == 1
        avg_ls = losssum()
        display("Loss: $(round(avg_ls, digits=5))")
    end
end

# NeuralFMU setup
numInputs = length(myFMU.modelDescription.inputValueReferences)
numOutputs = length(myFMU.modelDescription.outputValueReferences)

net = Chain(inputs -> fmi2InputDoStepCSOutput(myFMU, t_step, inputs),
            Dense(numOutputs, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numOutputs))

problem = CS_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(), tData, true, true)
solutionBefore = problem(t_start, t_step, t_stop, extF)
fmiPlot(problem)

# train it ...
p_net = Flux.params(problem)

optim = ADAM()
Flux.train!(losssum, p_net, Iterators.repeated((), 300), optim; cb=callb) # Feel free to increase training steps or epochs for better results

###### plot results a
solutionAfter = problem(t_start, t_step, t_stop, extF)
fig = Plots.plot(xlabel="t [s]", ylabel="mass acceleration [m s^-2]", linewidth=2,
    xtickfontsize=12, ytickfontsize=12,
    xguidefontsize=12, yguidefontsize=12,
    legendfontsize=12, legend=:bottomright)
Plots.plot!(fig, tData, fmi2SimulationResultGetValues(fmuSimData, "mass.a"), label="FMU", linewidth=2)
Plots.plot!(fig, tData, accData, label="reference", linewidth=2)
Plots.plot!(fig, tData, collect(data[2] for data in solutionAfter), label="NeuralFMU", linewidth=2)
Plots.savefig(fig, "exampleResult_a.pdf")

fmiUnload(myFMU)
