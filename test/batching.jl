#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux

import Random
Random.seed!(1234);

t_start = 0.0
t_step = 0.01
t_stop = 50.0
tData = t_start:t_step:t_stop

# generate training data
posData, velData, accData = syntTrainingData(tData)

# load FMU for NeuralFMU
fmu = loadFMU("SpringPendulum1D", EXPORTINGTOOL, EXPORTINGVERSION; type = :ME)

using FMIFlux.FMIImport
using FMIFlux.FMIImport.FMICore

# loss function for training
losssum_single = function (p)
    global problem, X0, posData
    solution = problem(X0; p = p, showProgress = false, saveat = tData)

    if !solution.success
        return Inf
    end

    posNet = getState(solution, 1; isIndex = true)

    return Flux.Losses.mse(posNet, posData)
end

numStates = length(fmu.modelDescription.stateValueReferences)

c1 = CacheLayer()
c2 = CacheRetrieveLayer(c1)

# the "Chain" for training
net = Chain(
    x -> fmu(; x = x, dx_refs = :all),
    dx -> c1(dx),
    Dense(numStates, 12, tanh),
    Dense(12, 1, identity),
    dx -> c2(1, dx[1]),
)

solver = Tsit5()
problem = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver; saveat = tData)
@test problem != nothing

function getInitialState(t)
    [syntTrainingData(t)[1:2]...]
end

train_data = collect([u] for u in posData)

c.fmu.executionConfig = deepcopy(c.fmu.executionConfig)
c.fmu.executionConfig.freeInstance = false 
c.fmu.executionConfig.instantiate = false
c.fmu.executionConfig.setup = true
c.fmu.executionConfig.reset = true
c.fmu.executionConfig.terminate = true
c, _ = FMIFlux.prepareSolveFMU(
    fmu,
    nothing,
    fmu.type;
    instantiate=true,
    setup=true
)

# batching 
batch = batchDataSolution(
    problem, 
    getInitialState, 
    tData,
    train_data; 
    batchDuration = 1.0,
    indicesModel = [1],
    plot = false, 
    showProgress = true,
)

len = length(batch) 
for i in 1:len
    b_tstart = batch[i].tStart
    b_tstop = batch[i].tStop
    b_saveat = b_tstart:0.01:b_tstop

    if i > 1 
        @test b_tstart == batch[i-1].tStop
    end
    if i < len
        @test b_tstop == batch[i+1].tStart
    end

    #solution = FMIFlux.run!(problem, batch[i]; saveat=b_saveat)
    #sim = collect(u[1] for u in solution.states.u)

    data = collect(getInitialState(t)[1] for t in b_saveat)
    targets = collect(tar[1] for tar in batch[i].targets)
    loss = Flux.Losses.mae(data, targets)
    @test loss < 1e-8
end

unloadFMU(fmu)
