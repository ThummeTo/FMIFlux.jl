# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

# imports
using FMI                       # for importing and simulating FMUs
using FMIFlux                   # for building neural FMUs
using FMIFlux.Flux              # the default machine learning library in Julia
using FMIZoo                    # a collection of demo FMUs
using Plots                     # for plotting some results

import Random                   # for random variables (and random initialization)
Random.seed!(1234)              # makes our program deterministic

tStart = 0.0
tStep = 0.01
tStop = 5.0
tSave = collect(tStart:tStep:tStop)

# let's load the FMU in CS-mode (some FMUs support multiple simulation modes)
fmu_gt = loadFMU("SpringPendulum1D", "Dymola", "2022x"; type=:CS)  

# and print some info
info(fmu_gt)   

# the initial state we start our simulation with, position (0.5 m) and velocity (0.0 m/s) of the pendulum
x0 = [0.5, 0.0] 

# some variables we are interested in, so let's record them: position, velocity and acceleration
vrs = ["mass.s", "mass.v", "mass.a"]  

# set the start state via parameters 
parameters = Dict("mass_s0" => x0[1], "mass_v0" => x0[2]) 

# simulate the FMU ...
sol_gt = simulate(fmu_gt, (tStart, tStop); recordValues=vrs, saveat=tSave, parameters=parameters)    

# ... and plot it!
plot(sol_gt)                                                                    

vel_gt = getValue(sol_gt, "mass.v")
acc_gt = getValue(sol_gt, "mass.a")

unloadFMU(fmu_gt)

fmu = loadFMU("SpringPendulumExtForce1D", "Dymola", "2022x"; type=:CS)
info(fmu)

# set the start state via parameters 
parameters = Dict("mass_s0" => x0[1], "mass.v" => x0[2])

sol_fmu = simulate(fmu, (tStart, tStop); recordValues=vrs, saveat=tSave, parameters=parameters)
plot(sol_fmu)

# outputs
println("Outputs:")
y_refs = fmu.modelDescription.outputValueReferences 
numOutputs = length(y_refs)
for y_ref in y_refs 
    name = valueReferenceToString(fmu, y_ref)
    println("$(y_ref) -> $(name)")
end

# inputs
println("\nInputs:")
u_refs = fmu.modelDescription.inputValueReferences 
numInputs = length(u_refs)
for u_ref in u_refs 
    name = valueReferenceToString(fmu, u_ref)
    println("$(u_ref) -> $(name)")
end

net = Chain(u -> fmu(;u_refs=u_refs, u=u, y_refs=y_refs),   # we can use the FMU just like any other neural network layer!
            Dense(numOutputs, 16, tanh),                    # some additional dense layers ...
            Dense(16, 16, tanh),
            Dense(16, numOutputs))

# the neural FMU is constructed by providing the FMU, the net topology, start and stop time
neuralFMU = CS_NeuralFMU(fmu, net, (tStart, tStop));

function extForce(t)
    return [0.0]
end 

solutionBefore = neuralFMU(extForce, tStep, (tStart, tStop); parameters=parameters)
plot(solutionBefore)

plot!(sol_gt)

function loss(p)
    # simulate the neural FMU by calling it
    sol_nfmu = neuralFMU(extForce, tStep, (tStart, tStop); parameters=parameters, p=p)

    # we use the second value, because we know that's the acceleration
    acc_nfmu = getValue(sol_nfmu, 2; isIndex=true)
    
    # we could also identify the position state by its name
    #acc_nfmu = getValue(sol_nfmu, "mass.a")
    
    FMIFlux.Losses.mse(acc_gt, acc_nfmu) 
end

global counter = 0
function callback(p)
    global counter += 1
    if counter % 20 == 1
        lossVal = loss(p[1])
        @info "Loss [$(counter)]: $(round(lossVal, digits=6))"
    end
end

optim = Adam()

p = FMIFlux.params(neuralFMU)

FMIFlux.train!(
    loss, 
    neuralFMU,
    Iterators.repeated((), 500), 
    optim; 
    cb=()->callback(p)
) 

solutionAfter = neuralFMU(extForce, tStep, (tStart, tStop); parameters=parameters)

fig = plot(solutionBefore; valueIndices=2:2, label="Neural FMU (before)", ylabel="acceleration [m/s^2]")
plot!(fig, solutionAfter; valueIndices=2:2, label="Neural FMU (after)")
plot!(fig, tSave, acc_gt; label="ground truth")
fig

unloadFMU(fmu)

# check package build information for reproducibility
import Pkg; Pkg.status()
