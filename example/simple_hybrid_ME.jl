#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

################################## INSTALLATION ####################################################
# (1) Enter Package Manager via     ]
# (2) Install FMI via               add FMI       or   add "https://github.com/ThummeTo/FMI.jl"
# (3) Install FMIFlux via           add FMIFlux   or   add "https://github.com/ThummeTo/FMIFlux.jl"
################################ END INSTALLATION ##################################################

# this example covers creation and training am ME-NeuralFMUs

using FMI
using FMIFlux
using Flux
using DifferentialEquations: Tsit5
import Plots

modelFMUPath = joinpath(dirname(@__FILE__), "../model/SpringPendulum1D.fmu")
realFMUPath = joinpath(dirname(@__FILE__), "../model/SpringFrictionPendulum1D.fmu")

t_start = 0.0
t_step = 0.01
t_stop = 5.0

myFMU = fmiLoad(realFMUPath)
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)

fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

x0 = fmi2GetContinuousStates(myFMU)

realSimData = fmi2SimulateCS(myFMU, t_step, t_start, t_stop, ["mass.s", "mass.v", "mass.f", "mass.a"], false)
fmiUnload(myFMU)

fmiPlot(realSimData)

myFMU = fmiLoad(modelFMUPath)

fmiInstantiate!(myFMU; loggingOn=false)
fmuSimData = fmiSimulate(myFMU, t_step, t_start, t_stop, ["mass.s", "mass.v", "mass.a"])

tData = t_start:t_step:t_stop
posData = fmi2SimulationResultGetValues(realSimData, "mass.s")
velData = fmi2SimulationResultGetValues(realSimData, "mass.v")

# loss function for training
function losssum()
    solution = problem(t_start, x0)

    posNet = collect(data[2] for data in solution.u)
    velNet = collect(data[3] for data in solution.u)

    Flux.Losses.mse(posNet, posData) + Flux.Losses.mse(velNet, velData)
end

# callback function for training
global iterCB = 0
function callb()
    global iterCB += 1

    if iterCB % 10 == 1
        avg_ls = losssum()
        display("Loss: $(round(avg_ls, digits=5))   Avg displacement in data: $(round(sqrt(avg_ls), digits=5))")
    end
end

# NeuralFMU setup
numStates = fmiGetNumberOfStates(myFMU)

net = Chain(inputs -> fmi2DoStepME(myFMU, inputs),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))

problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(), tData)
solutionBefore = problem(t_start, x0)
fmiPlot(problem)

# train it ...
p_net = Flux.params(problem)

optim = ADAM()
Flux.train!(losssum, p_net, Iterators.repeated((), 1000), optim; cb=callb) # Feel free to increase training steps or epochs for better results

###### plot results mass.s
solutionAfter = problem(t_start, x0)
fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
    xtickfontsize=12, ytickfontsize=12,
    xguidefontsize=12, yguidefontsize=12,
    legendfontsize=12, legend=:bottomright)
Plots.plot!(fig, tData, fmi2SimulationResultGetValues(fmuSimData, "mass.s"), label="FMU", linewidth=2)
Plots.plot!(fig, tData, posData, label="reference", linewidth=2)
Plots.plot!(fig, tData, collect(data[2] for data in solutionAfter.u), label="NeuralFMU", linewidth=2)
Plots.savefig(fig, "exampleResult_s.pdf")

fmiUnload(myFMU)
