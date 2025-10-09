#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using Statistics
using DifferentialEquations
using FMIFlux, FMIZoo, Test
import FMIFlux.FMISensitivity.SciMLSensitivity.SciMLBase: RightRootFind, LeftRootFind
import FMIFlux.FMIImport.FMIBase: unsense
using FMIFlux.FMISensitivity.SciMLSensitivity.ForwardDiff,
    FMIFlux.FMISensitivity.SciMLSensitivity.ReverseDiff,
    FMIFlux.FMISensitivity.SciMLSensitivity.FiniteDiff,
    FMIFlux.FMISensitivity.SciMLSensitivity.Zygote
using FMIFlux.FMIImport, FMIFlux.FMIImport.FMICore, FMIZoo
import LinearAlgebra: I
import FMIFlux: ReverseDiffAdjoint

import Random
Random.seed!(5678);

global solution = nothing
global events = 0

DAMPING = 0.7
RADIUS = 0.0
GRAVITY = 9.81
DBL_MIN = 1e-10

NUMEVENTS = 4

t_start = 0.0
t_step = 0.05
t_stop = 2.0
tData = t_start:t_step:t_stop
posData = ones(Float64, length(tData))
x0_bb = [1.0, 0.0]

solvekwargs =
    Dict{Symbol,Any}(:saveat => tData, :abstol => 1e-6, :reltol => 1e-6, :dtmax => 1e-2)

numStates = 2
solver = Tsit5()
atols = 1e-3

Wr = rand(2, 2) * 1e-4 # zeros(2,2) # 
br = rand(2) * 1e-4 # zeros(2) #
W1 = [1.0 0.0; 0.0 1.0] - Wr
b1 = [0.0, 0.0] - br
W2 = [1.0 0.0; 0.0 1.0] - Wr
b2 = [0.0, 0.0] - br

fmu = loadFMU("BouncingBall1D", "Dymola", "2023x"; type = :ME, logLevel=:info)
fmu_params = Dict("damping" => DAMPING, "mass_radius" => RADIUS, "mass_s_min" => DBL_MIN)
sensealg = ReverseDiffAdjoint()

loss = function(p; kwargs...)
    global solution
    solution = prob(
        x0_bb;
        p = p,
        alg = solver,
        parameters = fmu_params,
        sensealg = sensealg,
        kwargs...,
        solvekwargs...,
    )

    return sum(abs.(collect(u[1] for u in solution.states.u)))
end

net = Chain(#Dense(W1, b1, identity),
    x -> fmu(; x = x, dx_refs = :all),
    Dense(W2, b2, identity),
)
prob = ME_NeuralFMU(fmu, net, (t_start, t_stop))
p = FMIFlux.params(prob)

# prepare
prob.snapshots = false
instantiate = fmu.executionConfig.instantiate
setup = fmu.executionConfig.setup
freeInstance = fmu.executionConfig.freeInstance

loss(p)
c = getCurrentInstance(fmu)

# add one persisitent snapshot
snapshot!(c)
@test length(c.snapshots) == 1

# if no snapshots wanted, and no snapshots required
prob.snapshots = false
loss(p)
c = getCurrentInstance(fmu)
@test length(c.snapshots) == 0
@test length(c.solution.snapshots) == 0

# if snapshots wanted, and snapshots required (free instance = true)
prob.snapshots = true
loss(p)
c = getCurrentInstance(fmu)
@test length(c.snapshots) == NUMEVENTS*2
@test length(c.solution.events) == NUMEVENTS
@test length(c.solution.snapshots) == NUMEVENTS*2

# now we switcht to single instance
fmu.executionConfig.freeInstance = false
fmu.executionConfig.setup = false
fmu.executionConfig.instantiate = false

prob.snapshots = false
loss(p)
c = getCurrentInstance(fmu)
addr = c.addr

# add a single persisitent snapshot
snapshot!(c)
@test length(c.snapshots) == 1

prob.snapshots = true
loss(p)
@test c.addr == addr
@test length(c.snapshots) == 1+NUMEVENTS*2
@test length(c.solution.events) == NUMEVENTS
@test length(c.solution.snapshots) == NUMEVENTS*2

loss(p)
@test length(c.snapshots) == 1+NUMEVENTS*4
@test length(c.solution.snapshots) == NUMEVENTS*2

loss(p; cleanSnapshots=true)
@test length(c.snapshots) == 1+NUMEVENTS*6
@test length(c.solution.snapshots) == 0

# reset 
fmu.executionConfig.instantiate = instantiate
fmu.executionConfig.setup = setup
fmu.executionConfig.freeInstance = freeInstance

# small   = fmu.executionConfig.snapshotDeltaTimeTolerance * 1e-2     # deviation that maps to the same snapshot
# big     = fmu.executionConfig.snapshotDeltaTimeTolerance * 1e2      # deviation that maps to the previous/next snapshot
# for i in 1:NUMEVENTS
#     local s, t 
    
#     t = solution.events[i].t

#     @info "Event $(i) @ t=$(t)s"

#     # get snapshot at event (little deviation)
#     s = getSnapshot(c, t-small)
#     @test s == solution.snapshots[i+1]
#     s = getPreviousSnapshot(c, t-small)
#     @test s == solution.snapshots[i]

#     # get snapshot at the event
#     s = getSnapshot(c, t)
#     @test s == solution.snapshots[i+1] 
#     s = getPreviousSnapshot(c, t)
#     @test s == solution.snapshots[i]

#     # get snapshot slightly after event
#     s = getSnapshot(c, t+small)
#     @test s == solution.snapshots[i+1]     
#     s = getPreviousSnapshot(c, t+small)
#     @test s == solution.snapshots[i]

#     # try to get snapshot after the event (out of range)
#     s = getSnapshot(c, t+big)
#     @test s == nothing
#     s = getPreviousSnapshot(c, t+big)
#     @test s == solution.snapshots[i+1]

#     # try to get snapshot before the event (out of range)
#     s = getSnapshot(c, t-big)
#     @test s == nothing
#     s = getPreviousSnapshot(c, t-big)
#     @test s == solution.snapshots[i]
# end

# # test for super-dense time
# s1 = snapshot!(c)
# s2 = snapshot!(c)

# @test s1.t == s2.t == t_stop
# @test s1.index == 0
# @test s2.index == 1

unloadFMU(fmu)
