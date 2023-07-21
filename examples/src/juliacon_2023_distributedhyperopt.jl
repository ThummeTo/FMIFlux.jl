# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

# This workshop was held at the JuliaCon2023 @ MIT (Boston)

using Plots
using Distributed
using JLD2
using DistributedHyperOpt

# if you want to see more messages about Hyperband working ...
# ENV["JULIA_DEBUG"] = "DistributedHyperOpt"

nprocs()
workers = addprocs(5) 
@everywhere include(joinpath(@__DIR__, "workshop_module.jl"))

# set the current hyperparameter optimization run
@everywhere NODE_Training.HPRUN = 1
@info "Run: $(NODE_Training.HPRUN)"

# creating paths for log files (logs), parameter sets (params) and hyperparameter plots (plots) 
for dir ∈ ("logs", "params", "plots", "results") 
    path = joinpath(@__DIR__, dir, "$(NODE_Training.HPRUN)")
    @info "Creating (if not already) path: $(path)"
    mkpath(path)
end 

beta1 = 1.0 .- exp10.(LinRange(-4,-1,4))
beta2 = 1.0 .- exp10.(LinRange(-6,-1,6))

sampler = DistributedHyperOpt.Hyperband(;R=81, η=3, ressourceScale=1.0/81.0*NODE_Training.data.cumconsumption_t[end])
optimization = DistributedHyperOpt.Optimization(NODE_Training.train!, 
                                             DistributedHyperOpt.Parameter("eta", (1e-5, 1e-2); type=:Log, samples=7, round_digits=5),  
                                             DistributedHyperOpt.Parameter("beta1", beta1), 
                                             DistributedHyperOpt.Parameter("beta2", beta2), 
                                             DistributedHyperOpt.Parameter("batchDur", (0.5, 20.0); samples=40, round_digits=1), 
                                             DistributedHyperOpt.Parameter("lastWeight", (0.1, 1.0); samples=10, round_digits=1),
                                             DistributedHyperOpt.Parameter("schedulerID", [:Random, :Sequential, :LossAccumulation]),
                                             DistributedHyperOpt.Parameter("loss", [:MSE, :MAE]) )
DistributedHyperOpt.optimize(optimization; 
                          sampler=sampler, 
                          plot=true, 
                          plot_ressources=true,
                          save_plot=joinpath(@__DIR__, "plots", "$(NODE_Training.HPRUN)", "hyperoptim.png"),
                          redirect_worker_io_dir=joinpath(@__DIR__, "logs", "$(NODE_Training.HPRUN)"))

Plots.plot(optimization; size=(1024, 1024), ressources=true)
minimum, minimizer, ressource = DistributedHyperOpt.results(optimization)
