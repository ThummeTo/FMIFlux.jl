#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

################################## INSTALLATION ####################################################
# (1) Enter Package Manager via     ]
# (2) Install FMI via               add FMI       or   add "https://github.com/ThummeTo/FMI.jl"
# (3) Install FMIFlux via           add FMIFlux   or   add "https://github.com/ThummeTo/FMIFlux.jl"
################################ END INSTALLATION ##################################################

# this example covers creation and training of a ME-NeuralFMU

using FMI
using FMIFlux
using Flux
using DifferentialEquations: Tsit5
import Plots
using Zygote
using FMIZoo

t_start = 0.0
t_step = 0.1
t_stop = 5.0
tData = collect(t_start:t_step:t_stop)

myFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)

fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

x0 = fmi2GetContinuousStates(myFMU)

vrs = ["mass.s", "mass.v", "mass.f", "mass.a"]
success, realSimData = fmiSimulate(myFMU, t_start, t_stop; saveat=tData, recordValues=["mass.s", "mass.v", "mass.f", "mass.a"], setup=false, reset=false)
fmiUnload(myFMU)

fmiPlot(myFMU, vrs, realSimData)

myFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")

fmiInstantiate!(myFMU; loggingOn=false)
_, fmuSimData = fmiSimulate(myFMU, t_start, t_stop; saveat=tData, recordValues=["mass.s", "mass.v", "mass.a"])

posData = collect(data[1] for data in realSimData.saveval)
velData = collect(data[2] for data in realSimData.saveval)

# loss function for training
function losssum()
    global integratorSteps, problem

    solution = problem(x0)

    posNet = collect(data[1] for data in solution.u)
    # velNet = collect(data[2] for data in solution.u)

    Flux.mse(posData, posNet)
end

# callback function for training
global iterCB = 0
function callb()
    global iterCB += 1
   
    if iterCB % 30 == 1
        avg_ls = losssum()
        @info "Loss [$iterCB]: $(round(avg_ls, digits=5))   Avg displacement in data: $(round(sqrt(avg_ls), digits=5))"
   
        fig = plotResults()
        println("Fig. update.")
        display(fig)
    end
end

function plotResults()
    solutionAfter = problem(x0, t_start)
    fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12, legend=:bottomright)
    Plots.plot!(fig, tData, collect(data[1] for data in fmuSimData.saveval), label="FMU", linewidth=2)
    Plots.plot!(fig, tData, posData, label="reference", linewidth=2)
    Plots.plot!(fig, solutionAfter.t, collect(data[1] for data in solutionAfter.u), label="NeuralFMU", linewidth=2)
    fig
end

# NeuralFMU setup
numStates = fmiGetNumberOfStates(myFMU)
additionalVRs = [fmi2String2ValueReference(myFMU, "mass.m")]
numAdditionalVRs = length(additionalVRs)

net = Chain(inputs -> fmiDoStepME(myFMU, inputs, -1.0, zeros(fmi2ValueReference, 0), zeros(fmi2Real, 0), additionalVRs), 
            Dense(numStates+numAdditionalVRs, 16, tanh), 
            Dense(16, 16, tanh),
            Dense(16, numStates))

problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(); saveat=tData)
solutionBefore = problem(x0, t_start)
fmiPlot(myFMU, solutionBefore)

# train it ...
p_net = Flux.params(problem)

optim = ADAM()
# Feel free to increase training steps or epochs for better results
Flux.train!(losssum, p_net, Iterators.repeated((), 1000), optim; cb=callb)

fmiUnload(myFMU)
