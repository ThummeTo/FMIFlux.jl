# [Training a simple Hybrid CS FMU](@id simpleCS)

This example explains how to create and train a CS-Neural FMU.

First the necessary libraries are loaded and the FMU is prepared for simulation.

```julia
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

FMUPath = joinpath(dirname(@__FILE__), "..", "model", "SpringPendulumExtForce1D.fmu")

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

myFMU = fmiLoad(FMUPath)
fmiInstantiate!(myFMU; loggingOn=false)
```
Then the FMU is simulated twice. Once with an increased amplitude and inverted phase. This is used as the real Simulation data which the FMU should represent after training. During the simulation the data needed for training is collected.
```julia
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiSetReal(myFMU, "mass_s0", 1.3)   # increase amplitude, invert phase
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)



realSimData = fmiSimulate(myFMU, t_start, t_stop; recordValues=["mass.s", "mass.v", "mass.a"], setup=false, saveat=tData)
fmiPlot(realSimData)

fmiReset(myFMU)
fmiSetupExperiment(myFMU, t_start, t_stop)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

fmuSimData = fmiSimulate(myFMU, t_start, t_stop; recordValues=["mass.s", "mass.v", "mass.a"], setup=false, saveat=tData)
fmiPlot(fmuSimData)
```
Before the training can start the loss and callback functions are defined.
```julia
######

extF = zeros(length(tData)) # no external force
posData = fmi2SimulationResultGetValues(realSimData, "mass.s")
velData = fmi2SimulationResultGetValues(realSimData, "mass.v")
accData = fmi2SimulationResultGetValues(realSimData, "mass.a")

# loss function for training
function losssum()
    solution = problem(t_step; inputs=extF)

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
        @info "Loss: $(round(avg_ls, digits=5))"
    end
end
```
After that the net is created and trained.
```julia
# NeuralFMU setup
numInputs = length(myFMU.modelDescription.inputValueReferences)
numOutputs = length(myFMU.modelDescription.outputValueReferences)

net = Chain(inputs -> fmiInputDoStepCSOutput(myFMU, t_step, inputs),
            Dense(numOutputs, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numOutputs))

problem = CS_NeuralFMU(myFMU, net, (t_start, t_stop); saveat=tData)
solutionBefore = problem(t_step; inputs=extF)

# train it ...
p_net = Flux.params(problem)

optim = ADAM()
Flux.train!(losssum, p_net, Iterators.repeated((), 300), optim; cb=callb) # Feel free to increase training steps or epochs for better results
```
And the results are plotted.
```julia
###### plot results a
solutionAfter = problem(t_step; inputs=extF)
fig = Plots.plot(xlabel="t [s]", ylabel="mass acceleration [m s^-2]", linewidth=2,
    xtickfontsize=12, ytickfontsize=12,
    xguidefontsize=12, yguidefontsize=12,
    legendfontsize=12, legend=:bottomright)
Plots.plot!(fig, tData, fmi2SimulationResultGetValues(fmuSimData, "mass.a"), label="FMU", linewidth=2)
Plots.plot!(fig, tData, accData, label="reference", linewidth=2)
Plots.plot!(fig, tData, collect(data[2] for data in solutionAfter), label="NeuralFMU", linewidth=2)
Plots.savefig(fig, "exampleResult_a.pdf")

fmiUnload(myFMU)
```