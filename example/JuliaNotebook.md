# This example covers creation and training of ME-NeuralFMUs

## Motivation
This Julia Package is motivated by the application of hybrid modeling. This package enables the user to integrate his simulation model between neural networks (NeuralFMU). For this, the simulation model must be exported as FMU (functional mock-up unit), which corresponds to a widely used standard. The big advantage of a hybrid modeling with artificial neural networks is, that effects, that are difficult to model, can easily be learned by the neural networks. For this purpose, the so-called NeuralFMU is trained with measurement data containing the effect and as a final product a simulation with the mapping of complex effects is obtained. Another big advantage of the NeuralFMU is that it works with relatively little data, because the FMU already contains the rough functionality of the simulation and only the missing effects are added.

## Introduction to the example
Test!!!!!!!!!!!!!!


Grob Erklärung der Funktion
Zielgruppe: Leute, die Modell bauen und Machine Learning mit diesen Modell durchführen wollen
            -> hybride Modellbildung


Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons

Licensed under the MIT license. See LICENSE file in the project root for details.

## Installation prerequisites

|    | Description                       | Command     |  Alternative  |   
|:--- |:---                               |:---:        |:---:|
|1.  | Enter Package Manager via         |     ]       |     |
|2.  | Install FMI via                   |   add FMI   | add " https://github.com/ThummeTo/FMI.jl "   |
|3.  | Install FMIFlux via               | add FMIFlux | add " https://github.com/ThummeTo/FMIFlux.jl " |
|4.  | Install Flux via                  |  add Flux   |     |
|5.  | Install DifferentialEquations via | add DifferentialEquations |  |
|6.  | Install Plots via                 | add Plots   |     |

## Code section

To run the example, the previously installed packages must be included. 


```julia
# imports
# using Pkg
# Pkg.activate("Your Env")
using FMI
using FMIFlux
using Flux
using DifferentialEquations: Tsit5
import Plots
```

After importing the packages, the path to the FMUs (FMU = functional mock-up unit) is set. The FMU is a model from the Functional Mock-up Interface (FMI) Standard. The FMI is a free standard that defines a container and an interface to exchange dynamic models using a combination of XML files, binaries and C code zipped into a single file. Here the path for the [*SpringPendulum1D*](../model/SpringPendulum1D.mo) and the *SpringFrictionPendulum1D* model is set. The structure of the *SpringPendulum1D* can be seen in the following graphic and corresponds to a simple modeling.

<img src="./pics/SpringPendulum1D.png" alt="" width="300"/>


In contrast, the other model (*SpringFrictionPendulum1D*) is somewhat more accurate, because it includes a friction component. 

<img src="./pics/SpringFrictionPendulum1D.png" alt="" width="300"/>


```julia
modelFMUPath = joinpath(dirname(@__FILE__), "../model/SpringPendulum1D.fmu")
realFMUPath = joinpath(dirname(@__FILE__), "../model/SpringFrictionPendulum1D.fmu")
```






```julia

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = collect(t_start:t_step:t_stop)

myFMU = fmiLoad(realFMUPath)
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)

fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

x0 = fmiGetContinuousStates(myFMU)

vrs = ["mass.s", "mass.v", "mass.f", "mass.a"]
_, realSimData = fmiSimulate(myFMU, t_start, t_stop; recordValues=vrs, saveat=tData, setup=false, reset=false)
fmiUnload(myFMU)

fmiPlot(myFMU, vrs, realSimData)

myFMU = fmiLoad(modelFMUPath)

fmiInstantiate!(myFMU; loggingOn=false)
_, fmuSimData = fmiSimulate(myFMU, t_start, t_stop; recordValues=["mass.s", "mass.v", "mass.a"], saveat=tData)

posData = collect(data[1] for data in realSimData.saveval)
velData = collect(data[2] for data in realSimData.saveval)

# loss function for training
function losssum()
    solution = problem(x0, t_start)

    posNet = collect(data[1] for data in solution.u)
    #velNet = collect(data[2] for data in solution.u)

    Flux.Losses.mse(posData, posNet) #+ Flux.Losses.mse(velData, velNet)
end

# callback function for training
global iterCB = 0
function callb()
    global iterCB += 1

    if iterCB % 10 == 1
        avg_ls = losssum()
        @info "Loss [$iterCB]: $(round(avg_ls, digits=5))   Avg displacement in data: $(round(sqrt(avg_ls), digits=5))"
    end
end

# NeuralFMU setup
numStates = fmiGetNumberOfStates(myFMU)

net = Chain(inputs -> fmiDoStepME(myFMU, inputs),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))

problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(); saveat=tData)
solutionBefore = problem(x0, t_start)
fmiPlot(myFMU, solutionBefore)

# train it ...
p_net = Flux.params(problem)

optim = ADAM()
Flux.train!(losssum, p_net, Iterators.repeated((), 300), optim; cb=callb) # Feel free to increase training steps or epochs for better results

###### plot results mass.s
solutionAfter = problem(x0, t_start)
fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
    xtickfontsize=12, ytickfontsize=12,
    xguidefontsize=12, yguidefontsize=12,
    legendfontsize=12, legend=:bottomright)
Plots.plot!(fig, tData, collect(data[1] for data in fmuSimData.saveval), label="FMU", linewidth=2)
Plots.plot!(fig, tData, posData, label="reference", linewidth=2)
Plots.plot!(fig, tData, collect(data[2] for data in solutionAfter.u), label="NeuralFMU", linewidth=2)
Plots.savefig(fig, "exampleResult_s.pdf")
fig 

fmiUnload(myFMU)

```
