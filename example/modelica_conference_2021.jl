#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

################################## INSTALLATION ##################################
# (1) Enter Package Manager via     ]
# (2) Install FMI via               add "https://github.com/ThummeTo/FMI.jl"
# (3) Install FMIFlux via           add "https://github.com/ThummeTo/FMIFlux.jl"
################################ END INSTALLATION ################################

using FMI
using FMIFlux
using Flux
using DifferentialEquations: Tsit5
import Plots

modelFMUPath = joinpath(dirname(@__FILE__), "../model/SpringPendulum1D.fmu")
realFMUPath = joinpath(dirname(@__FILE__), "../model/SpringFrictionPendulum1D.fmu")

t_start = 0.0
t_step = 0.01
t_stop = 4.0
tData = t_start:t_step:t_stop

myFMU = fmiLoad(realFMUPath)
fmiInstantiate!(myFMU; loggingOn=false)

fmiReset(myFMU)
fmiSetupExperiment(myFMU, t_start, t_stop)
x0 = [0.5, 0.0]
fmiSetReal(myFMU, ["s0", "v0"], x0)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)
vrs = ["mass.s", "mass.v", "mass.a", "mass.f"]
success, realSimData = fmi2Simulate(myFMU, t_start, t_stop; recordValues=vrs, saveat=tData, setup=false, reset=false)
posData = collect(data[1] for data in realSimData.saveval)
velData = collect(data[2] for data in realSimData.saveval)

fmiReset(myFMU)
fmiSetupExperiment(myFMU, t_start, t_stop)
x0_test = [1.0, -1.5]
fmiSetReal(myFMU, ["s0", "v0"], x0_test)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)
success, realSimData_test = fmi2Simulate(myFMU, t_start, t_stop; recordValues=vrs, saveat=tData, setup=false, reset=false)
posData_test = collect(data[1] for data in realSimData_test.saveval)
velData_test = collect(data[2] for data in realSimData_test.saveval)

fmiPlot(myFMU, vrs, realSimData)
fmiPlot(myFMU, vrs, realSimData_test)

fmiUnload(myFMU)

displacement = 0.1
myFMU = fmiLoad(modelFMUPath)

fmiInstantiate!(myFMU; loggingOn=false)

# pure FMU simulation data (train)
fmiReset(myFMU)
fmiSetupExperiment(myFMU, 0.0)
fmiSetReal(myFMU, ["mass_s0", "mass_v0"], x0)
fmi2SetReal(myFMU, "fixed.s0", displacement)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)
_, fmuSimData = fmiSimulate(myFMU, t_start, t_stop; recordValues=["mass.s", "mass.v", "mass.a"], saveat=tData, setup=false, reset=false)

# pure FMU simulation data (test)
fmiReset(myFMU)
fmiSetupExperiment(myFMU, 0.0)
fmiSetReal(myFMU, ["mass_s0", "mass_v0"], x0_test)
fmi2SetReal(myFMU, "fixed.s0", displacement)
fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)
_, fmuSimData_test = fmiSimulate(myFMU, t_start, t_stop; recordValues=["mass.s", "mass.v", "mass.a"], saveat=tData, setup=false, reset=false)

# loss function for training
function losssum()
    global x0
    solution = problem(x0)

    posNet = collect(data[1] for data in solution.u)
    velNet = collect(data[2] for data in solution.u)

    (Flux.Losses.mse(posNet, posData) + Flux.Losses.mse(velNet, velData)) / 2.0
end

# callback function for training
global iterCB = 0
function callb()
    global iterCB, p_net
    iterCB += 1

    # freeze first layer parameters (2,4,6) for velocity -> (static) direct feed trough for velocity
    # parameters for position (1,3,5) are learned
    p_net[1][2] = 0.0
    p_net[1][4] = 1.0
    p_net[1][6] = 0.0

    if iterCB % 10 == 1
        avg_ls = losssum()
        @info "Loss [$iterCB]: $(round(avg_ls, digits=5))   Avg displacement in data: $(round(sqrt(avg_ls), digits=5))   Weight/Scale: $(p_net[1][1])   Bias/Offset: $(p_net[1][5])"
    end
end

# NeuralFMU setup
numStates = fmiGetNumberOfStates(myFMU)

net = Chain(Dense(numStates, numStates, identity; initW = (out, in) -> [[1.0, 0.0] [0.0, 1.0]], initb = out -> zeros(out)),
            inputs -> fmi2DoStepME(myFMU, inputs),
            Dense(numStates, 8),
            Dense(8, 8, tanh),
            Dense(8, numStates))

problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(); saveat=tData)
solutionBefore = problem(x0)

# train it ...
p_net = Flux.params(problem)

for i in 1:length(p_net[1])
    if p_net[1][i] < 0.0 
        p_net[1][i] = -p_net[1][i]
    end
end

optim = ADAM()
Flux.train!(losssum, p_net, Iterators.repeated((), 1), optim; cb=callb) # precompile Flux.train!

solutionAfter = []
solutionAfter_test = []

disp_s = []
fs = []

linestyles = [:dot, :solid] #, :dash]

for run in 1:2
    
    @time for i in 1:5
        @info "epoch: $i/5"
        Flux.train!(losssum, p_net, Iterators.repeated((), 500), optim; cb=callb)
    end
    push!(solutionAfter, problem(x0))
    push!(solutionAfter_test, problem(x0_test))

    ###### plot results s (training data)
    fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12, legend=:topright)
    Plots.plot!(fig, tData, collect(data[1] for data in fmuSimData.saveval), label="FMU", linewidth=2)
    Plots.plot!(fig, tData, posData, label="reference", linewidth=2)
    for s in 1:length(solutionAfter)
        Plots.plot!(fig, solutionAfter[s].t, collect(data[1] for data in solutionAfter[s].u), label="NeuralFMU ($(s*2500))", linewidth=2, linestyle=linestyles[s], linecolor=:green)
    end
    Plots.savefig(fig, "exampleResult_s_train$(run).pdf")

    ###### plot results s (testing data)
    fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12, legend=:topright)
    Plots.plot!(fig, tData, collect(data[1] for data in fmuSimData_test.saveval), label="FMU", linewidth=2)
    Plots.plot!(fig, tData, posData_test, label="reference", linewidth=2)
    for s in 1:length(solutionAfter)
        Plots.plot!(fig, solutionAfter_test[s].t, collect(data[1] for data in solutionAfter_test[s].u), label="NeuralFMU ($(s*2500))", linewidth=2, linestyle=linestyles[s], linecolor=:green)
    end
    Plots.savefig(fig, "exampleResult_s_test$(run).pdf")

    ###### plot results v (training data)
    fig = Plots.plot(xlabel="t [s]", ylabel="mass velocity [m/s]", linewidth=2,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12, legend=:topright)
    Plots.plot!(fig, tData, collect(data[2] for data in fmuSimData.saveval), label="FMU", linewidth=2)
    Plots.plot!(fig, tData, velData, label="reference", linewidth=2)
    for s in 1:length(solutionAfter)
        Plots.plot!(fig, solutionAfter[s].t, collect(data[2] for data in solutionAfter[s].u), label="NeuralFMU ($(s*2500))", linewidth=2, linestyle=linestyles[s], linecolor=:green)
    end
    Plots.savefig(fig, "exampleResult_v_train$(run).pdf")

    ###### plot results v (testing data)
    fig = Plots.plot(xlabel="t [s]", ylabel="mass velocity [m/s]", linewidth=2,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12, legend=:topright)
    Plots.plot!(fig, tData, collect(data[2] for data in fmuSimData_test.saveval), label="FMU", linewidth=2)
    Plots.plot!(fig, tData, velData_test, label="reference", linewidth=2)
    for s in 1:length(solutionAfter)
        Plots.plot!(fig, solutionAfter_test[s].t, collect(data[2] for data in solutionAfter_test[s].u), label="NeuralFMU ($(s*2500))", linewidth=2, linestyle=linestyles[s], linecolor=:green)
    end
    Plots.savefig(fig, "exampleResult_v_test$(run).pdf")

    ###### friction model extraction

    layers_bottom = problem.neuralODE.model.layers[3:5]
    net_bottom = Chain(layers_bottom...)
    transferParams!(net_bottom, p_net, 7)

    s_fmu = collect(data[1] for data in fmuSimData.saveval)
    v_fmu = collect(data[2] for data in fmuSimData.saveval)
    a_fmu = collect(data[3] for data in fmuSimData.saveval)

    s_real = collect(data[1] for data in realSimData.saveval)
    v_real = collect(data[2] for data in realSimData.saveval)
    a_real = collect(data[3] for data in realSimData.saveval)
    f_real = collect(data[4] for data in realSimData.saveval)

    push!(fs, zeros(length(v_real)))
    for i in 1:length(v_real)
        fs[run][i] = -net_bottom([v_real[i], 0.0])[2]
    end

    fig = Plots.plot(xlabel="v [m/s]", ylabel="friction force [N]", linewidth=2,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12, legend=:topright, xlim=(-1.25, 1.25))

    mat = hcat(v_real, zeros(length(v_real)))
    mat[sortperm(mat[:, 1]), :]
    Plots.plot!(fig, mat[:,1], mat[:,2], label="FMU", linewidth=2)

    mat_ref = hcat(v_real, f_real)
    mat_ref[sortperm(mat_ref[:, 1]), :]
    Plots.plot!(fig, mat_ref[:,1], mat_ref[:,2], label="reference", linewidth=2)

    for s in 1:length(fs)
        mat_neu = hcat(v_real, fs[s])
        mat_neu[sortperm(mat_neu[:, 1]), :]
        Plots.plot!(fig, mat_neu[:,1], mat_neu[:,2], label="NeuralFMU ($(s*2500))", linewidth=2, linestyle=linestyles[s], linecolor=:green)
        @info "Friction model $s mse: $(Flux.Losses.mse(mat_neu[:,2], mat_ref[:,2]))"
    end

    Plots.savefig(fig, "frictionModel$(run).pdf")

    #########

    layers_top = problem.neuralODE.model.layers[1:1]
    net_top = Chain(layers_top...)
    transferParams!(net_top, p_net, 1)

    push!(disp_s, zeros(length(s_real)))
    for i in 1:length(s_real)
        disp_s[run][i] = net_top([s_real[i], 0.0])[1] - s_real[i] - displacement
    end

    fig = Plots.plot(xlabel="t [s]", ylabel="displacement [m]", linewidth=2,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12, legend=:topright)

    Plots.plot!(fig, [t_start, t_stop], [displacement, displacement], label="FMU", linewidth=2)
    Plots.plot!(fig, [t_start, t_stop], [0.0, 0.0], label="reference", linewidth=2)
    for s in 1:length(disp_s)
        Plots.plot!(fig, tData, disp_s[s], label="NeuralFMU ($(s*2500))", linewidth=2, linestyle=linestyles[s], linecolor=:green)
    end

    Plots.savefig(fig, "displacementModel$(run).pdf")
end

fmiUnload(myFMU)
