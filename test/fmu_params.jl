#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using DifferentialEquations: Tsit5

import Random 
Random.seed!(1234);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
posData, velData, accData = syntTrainingData(tData)

# load FMU for NeuralFMU
# [TODO] Replace by a suitable discontinuous FMU
fmu = fmi2Load("SpringPendulum1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:ME)

using FMIFlux.FMIImport
using FMIFlux.FMIImport.FMICore

c = fmi2Instantiate!(fmu)
fmi2SetupExperiment(c, 0.0, 1.0)
fmi2EnterInitializationMode(c)
fmi2ExitInitializationMode(c)

p_refs = fmu.modelDescription.parameterValueReferences
p = fmi2GetReal(c, p_refs)

# loss function for training
losssum = function(p)
    #@info "$p"
    global problem, X0, posData, solution
    solution = problem(X0; p=p, showProgress=true, saveat=tData)

    if !solution.success
        return Inf 
    end

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    return Flux.Losses.mse(posNet, posData)
end

numStates = length(fmu.modelDescription.stateValueReferences)

# the "Chain" for training
net = Chain(FMUParameterRegistrator(fmu, p_refs, p),
            x -> fmu(x=x, dx_refs=:all)) # , fmuLayer(p))

optim = OPTIMISER(ETA)
solver = Tsit5()

problem = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver)
problem.modifiedState = false
@test problem != nothing

solutionBefore = problem(X0; saveat=tData)
@test solutionBefore.success
@test length(solutionBefore.states.t) == length(tData)
@test solutionBefore.states.t[1] == t_start
@test solutionBefore.states.t[end] == t_stop

# train it ...
p_net = Flux.params(problem)
@test length(p_net) == 1
@test length(p_net[1]) == 7
lossBefore = losssum(p_net[1])

@info "Start-Loss for net: $(lossBefore)"

# [ToDo] Discontinuous system?
# j_fin = FiniteDiff.finite_difference_gradient(losssum, p_net[1])
# j_fwd = ForwardDiff.gradient(losssum, p_net[1])
# j_rwd = ReverseDiff.gradient(losssum, p_net[1])

FMIFlux.train!(losssum, problem, Iterators.repeated((), NUMSTEPS), optim; gradient=GRADIENT)

# check results
solutionAfter = problem(X0; saveat=tData)
@test solutionAfter.success
@test length(solutionAfter.states.t) == length(tData)
@test solutionAfter.states.t[1] == t_start
@test solutionAfter.states.t[end] == t_stop

lossAfter = losssum(p_net[1])
@test lossAfter < lossBefore

@test length(fmu.components) <= 1

fmi2Unload(fmu)
