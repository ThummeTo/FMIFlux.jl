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
import FMIFlux: isimplicit

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

fmu = loadFMU("BouncingBall1D", "Dymola", "2023x"; type = :ME)
fmu_params = Dict("damping" => DAMPING, "mass_radius" => RADIUS, "mass_s_min" => DBL_MIN)
sensealg = ReverseDiffAdjoint()

net = Chain(#Dense(W1, b1, identity),
    x -> fmu(; x = x, dx_refs = :all),
    Dense(W2, b2, identity),
)
prob = ME_NeuralFMU(fmu, net, (t_start, t_stop))
prob.snapshots = true

fmu.executionConfig.freeInstance = true

solution = prob(
    x0_bb;
    p = p_net,
    solver = solver,
    parameters = fmu_params,
    sensealg = sensealg,
    cleanSnapshots = false,
    solvekwargs...,
)

c = getCurrentInstance(fmu)

# we need one snapshot more than events!
@test length(solution.events) == NUMEVENTS
@test length(solution.snapshots)-1 == NUMEVENTS

# check if the time instances are identically!
for i in 1:NUMEVENTS
    @test solution.snapshots[i+1].t == solution.events[i].t
end

EPS = 1e-8
for i in 1:NUMEVENTS
    t = solution.events[i].t

    # get snapshot slightly before the event
    s = getSnapshot(c, t-EPS)
    @test s == solution.snapshots[i]

    # get snapshot at the event
    s = getSnapshot(c, t)
    @test s == solution.snapshots[i]

    # get snapshot slightly after the event
    s = getSnapshot(c, t+EPS)
    @test s == solution.snapshots[min(i+1, NUMEVENTS+1)]
end

unloadFMU(fmu)
