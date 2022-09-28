# ME-NeuralFMU from the Modelica Conference 2021
Tutorial by Johannes Stoljar, Tobias Thummerer

## License
Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons, Johannes Stoljar

Licensed under the MIT license. See [LICENSE](https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

## Motivation
The Julia Package *FMIFlux.jl* is motivated by the application of hybrid modeling. This package enables the user to integrate his simulation model between neural networks (NeuralFMU). For this, the simulation model must be exported as FMU (functional mock-up unit), which corresponds to a widely used standard. The big advantage of hybrid modeling with artificial neural networks is, that the effects that are difficult to model (because they might be unknown) can be easily learned by the neural networks. For this purpose, the NeuralFMU is trained with measurement data containing the not modeled physical effect. The final product is a simulation model including the originally not modeled effects. Another big advantage of the NeuralFMU is that it works with little data, because the FMU already contains the characteristic functionality of the simulation and only the missing effects are added.

NeuralFMUs do not need to be as easy as in this example. Basically a NeuralFMU can combine different ANN topologies that manipulate any FMU-input (system state, system inputs, time) and any FMU-output (system state derivative, system outputs, other system variables). However, for this example a NeuralFMU topology as shown in the following picture is used.

![NeuralFMU.svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/pics/NeuralFMU.svg?raw=true)

*NeuralFMU (ME) from* [[1]](#Source).

## Introduction to the example
In this example, simplified modeling of a one-dimensional spring pendulum (without friction) is compared to a model of the same system that includes a nonlinear friction model. The FMU with the simplified model will be named *simpleFMU* in the following and the model with the friction will be named *realFMU*. At the beginning, the actual state of both simulations is shown, whereby clear deviations can be seen in the graphs. In addition, the initial states are changed for both models and these graphs are also contrasted, and the differences can again be clearly seen. The *realFMU* serves as a reference graph. The *simpleFMU* is then integrated into a NeuralFMU architecture and a training of the entire network is performed. After the training the final state is compared again to the *realFMU*. It can be clearly seen that by using the NeuralFMU, learning of the friction process has taken place.  


## Target group
The example is primarily intended for users who work in the field of first principle and/or hybrid modeling and are further interested in hybrid model building. The example wants to show how simple it is to combine FMUs with machine learning and to illustrate the advantages of this approach.


## Other formats
Besides, this [Jupyter Notebook](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/modelica_conference_2021.ipynb) there is also a [Julia file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/modelica_conference_2021.jl) with the same name, which contains only the code cells. For the documentation there is a [Markdown file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/modelica_conference_2021.md) corresponding to the notebook.  


## Getting started

### Installation prerequisites
|     | Description                       | Command                   | Alternative                                    |   
|:----|:----------------------------------|:--------------------------|:-----------------------------------------------|
| 1.  | Enter Package Manager via         | ]                         |                                                |
| 2.  | Install FMI via                   | add FMI                   | add " https://github.com/ThummeTo/FMI.jl "     |
| 3.  | Install FMIFlux via               | add FMIFlux               | add " https://github.com/ThummeTo/FMIFlux.jl " |
| 4.  | Install FMIZoo via                | add FMIZoo                | add " https://github.com/ThummeTo/FMIZoo.jl "  |
| 5.  | Install Flux via                  | add Flux                  |                                                |
| 6.  | Install DifferentialEquations via | add DifferentialEquations |                                                |
| 7.  | Install Plots via                 | add Plots                 |                                                |
| 8.  | Install Random via                | add Random                |                                                |

## Code section

To run the example, the previously installed packages must be included. 


```julia
# imports
using FMI
using FMIFlux
using FMIZoo
using Flux
using DifferentialEquations: Tsit5
import Plots

# set seed
import Random
Random.seed!(1234);
```

After importing the packages, the path to the *Functional Mock-up Units* (FMUs) is set. The exported FMU is a model meeting the *Functional Mock-up Interface* (FMI) Standard. The FMI is a free standard ([fmi-standard.org](http://fmi-standard.org/)) that defines a container and an interface to exchange dynamic models using a combination of XML files, binaries and C code zipped into a single file. 

The object-orientated structure of the *SpringPendulum1D* (*simpleFMU*) can be seen in the following graphic and corresponds to a simple modeling.

![svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/pics/SpringPendulum1D.svg?raw=true)

In contrast, the model *SpringFrictionPendulum1D* (*realFMU*) is somewhat more accurate, because it includes a friction component. 

![svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/pics/SpringFrictionPendulum1D.svg?raw=true)

Next, the start time and end time of the simulation are set. Finally, a step size is specified to store the results of the simulation at these time steps.


```julia
tStart = 0.0
tStep = 0.01
tStop = 4.0
tSave = collect(tStart:tStep:tStop)
```




    401-element Vector{Float64}:
     0.0
     0.01
     0.02
     0.03
     0.04
     0.05
     0.06
     0.07
     0.08
     0.09
     0.1
     0.11
     0.12
     ⋮
     3.89
     3.9
     3.91
     3.92
     3.93
     3.94
     3.95
     3.96
     3.97
     3.98
     3.99
     4.0



### RealFMU

In the next lines of code the FMU of the *realFMU* model from *FMIZoo.jl* is loaded and the information about the FMU is shown.  


```julia
realFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")
fmiInfo(realFMU)
```

    ┌ Info: fmi2Unzip(...): Successfully unzipped 153 files at `/tmp/fmijl_hqn46R/SpringFrictionPendulum1D`.
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/g4GUl/src/FMI2_ext.jl:76
    ┌ Info: fmi2Load(...): FMU resources location is `file:////tmp/fmijl_hqn46R/SpringFrictionPendulum1D/resources`
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/g4GUl/src/FMI2_ext.jl:192
    ┌ Info: fmi2Load(...): FMU supports both CS and ME, using CS as default if nothing specified.
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/g4GUl/src/FMI2_ext.jl:195


    #################### Begin information for FMU ####################
    	Model name:			SpringFrictionPendulum1D
    	FMI-Version:			2.0
    	GUID:				{2e178ad3-5e9b-48ec-a7b2-baa5669efc0c}
    	Generation tool:		Dymola Version 2022x (64-bit), 2021-10-08
    	Generation time:		2022-05-19T06:54:12Z
    	Var. naming conv.:		structured
    	Event indicators:		24
    	Inputs:				0
    	Outputs:			0
    	States:				2
    		33554432 ["mass.s"]
    		33554433 ["mass.v", "mass.v_relfric"]
    	Supports Co-Simulation:		true
    		Model identifier:	SpringFrictionPendulum1D
    		Get/Set State:		true
    		Serialize State:	true
    		Dir. Derivatives:	true
    		Var. com. steps:	true
    		Input interpol.:	true
    		Max order out. der.:	1
    	Supports Model-Exchange:	true
    		Model identifier:	SpringFrictionPendulum1D
    		Get/Set State:		true
    		Serialize State:	true
    		Dir. Derivatives:	true
    ##################### End information for FMU #####################


In the following two subsections, the *realFMU* is simulated twice with different initial states to show what effect the choice of initial states has.

#### Default initial states

In the next steps the parameters are defined. The first parameter is the initial position of the mass, which is initialized with $0.5m$, the second parameter is the initial velocity, which is initialized with $0\frac{m}{s}$. In the function `fmiSimulate()` the *realFMU* is simulated, still specifying the start and end time, the parameters and which variables are recorded. After the simulation is finished the result of the *realFMU* can be plotted. This plot also serves as a reference for the other model (*simpleFMU*). The extracted data will still be needed later on.


```julia
initStates = ["s0", "v0"]
x₀ = [0.5, 0.0]
params = Dict(zip(initStates, x₀))
vrs = ["mass.s", "mass.v", "mass.a", "mass.f"]

realSimData = fmiSimulate(realFMU, tStart, tStop; parameters=params, recordValues=vrs, saveat=tSave)
posReal = fmi2GetSolutionValue(realSimData, "mass.s")
velReal = fmi2GetSolutionValue(realSimData, "mass.v")
fmiPlot(realSimData)
```




    
![svg](modelica_conference_2021_files/modelica_conference_2021_10_0.svg)
    



#### Define functions

The structure of the previous code section is used more often in the further sections, so for clarity the previously explained code section for setting the paramters and simulating are combined into one function `simulate()`.


```julia
function simulate(FMU, initStates, x₀, variables, tStart, tStop, tSave)
    params = Dict(zip(initStates, x₀))
    return fmiSimulate(FMU, tStart, tStop; parameters=params, recordValues=variables, saveat=tSave)
end
```




    simulate (generic function with 1 method)



Also, a function to extract the position and velocity from the simulation data is created.


```julia
function extractPosVel(simData)
    if simData.states === nothing
        posData = fmi2GetSolutionValue(simData, "mass.s")
        velData = fmi2GetSolutionValue(simData, "mass.v")
    else
        posData = fmi2GetSolutionState(simData, 1; isIndex=true)
        velData = fmi2GetSolutionState(simData, 2; isIndex=true)
    end

    return posData, velData
end
```




    extractPosVel (generic function with 1 method)



#### Modified initial states

In contrast to the previous section, other initial states are selected. The position of the mass is initialized with $1.0m$ and the velocity is initialized with $-1.5\frac{m}{s}$. With the modified initial states the *realFMU* is simulated and a graph is generated.


```julia
xMod₀ = [1.0, -1.5]
realSimDataMod = simulate(realFMU, initStates, xMod₀, vrs, tStart, tStop, tSave)
fmiPlot(realSimDataMod)
```




    
![svg](modelica_conference_2021_files/modelica_conference_2021_16_0.svg)
    



 After the plots are created, the FMU is unloaded.


```julia
fmiUnload(realFMU)
```

### SimpleFMU

The following lines load the *simpleFMU* from *FMIZoo.jl*. 


```julia
simpleFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")
fmiInfo(simpleFMU)
```

    #################### Begin information for FMU ####################
    	Model name:			SpringPendulum1D
    	FMI-Version:			2.0
    	GUID:				{fc15d8c4-758b-48e6-b00e-5bf47b8b14e5}
    	Generation tool:		Dymola Version 2022x (64-bit), 2021-10-08
    	Generation time:		2022-05-19T06:54:23Z
    	Var. naming conv.:		structured
    	Event indicators:		0
    	Inputs:				0
    	Outputs:			0
    	States:				2
    		33554432 ["mass.s"]
    		33554433 ["mass.v"]
    	Supports Co-Simulation:		true
    		Model identifier:	SpringPendulum1D
    		Get/Set State:		true
    		Serialize State:	true
    		Dir. Derivatives:	true
    		Var. com. steps:	true
    		Input interpol.:	true
    		Max order out. der.:	1
    	Supports Model-Exchange:	true
    		Model identifier:	SpringPendulum1D
    		Get/Set State:		true
    		Serialize State:	true
    		Dir. Derivatives:	true
    ##################### End information for FMU #####################


    ┌ Info: fmi2Unzip(...): Successfully unzipped 153 files at `/tmp/fmijl_iz2bZR/SpringPendulum1D`.
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/g4GUl/src/FMI2_ext.jl:76
    ┌ Info: fmi2Load(...): FMU resources location is `file:////tmp/fmijl_iz2bZR/SpringPendulum1D/resources`
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/g4GUl/src/FMI2_ext.jl:192
    ┌ Info: fmi2Load(...): FMU supports both CS and ME, using CS as default if nothing specified.
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/g4GUl/src/FMI2_ext.jl:195


The differences between both systems can be clearly seen from the plots in the subchapters. In the plot for the *realFMU* it can be seen that the oscillation continues to decrease due to the effect of the friction. If you simulate long enough, the oscillation would come to a standstill in a certain time. The oscillation in the *simpleFMU* behaves differently, since the friction was not taken into account here. The oscillation in this model would continue to infinity with the same oscillation amplitude. From this observation the desire of an improvement of this model arises.     


In the following two subsections, the *simpleFMU* is simulated twice with different initial states to show what effect the choice of initial states has.

#### Default initial states

Similar to the simulation of the *realFMU*, the *simpleFMU* is also simulated with the default values for the position and velocity of the mass and then plotted. There is one difference, however, as another state representing a fixed displacement is set. In addition, the last variable is also removed from the variables to be plotted.


```julia
initStates = ["mass_s0", "mass_v0", "fixed.s0"]
displacement = 0.1
xSimple₀ = vcat(x₀, displacement)
vrs = vrs[1:end-1]

simpleSimData = simulate(simpleFMU, initStates, xSimple₀, vrs, tStart, tStop, tSave)
fmiPlot(simpleSimData)
```




    
![svg](modelica_conference_2021_files/modelica_conference_2021_23_0.svg)
    



#### Modified initial states

The same values for the initial states are used for this simulation as for the simulation from the *realFMU* with the modified initial states.


```julia
xSimpleMod₀ = vcat(xMod₀, displacement)

simpleSimDataMod = simulate(simpleFMU, initStates, xSimpleMod₀, vrs, tStart, tStop, tSave)
fmiPlot(simpleSimDataMod)
```




    
![svg](modelica_conference_2021_files/modelica_conference_2021_25_0.svg)
    



## NeuralFMU

#### Loss function

In order to train our model, a loss function must be implemented. The solver of the NeuralFMU can calculate the gradient of the loss function. The gradient descent is needed to adjust the weights in the neural network so that the sum of the error is reduced and the model becomes more accurate.

The error function in this implementation consists of the mean of the mean squared errors. The first part of the addition is the deviation of the position and the second part is the deviation of the velocity. The mean squared error (mse) for the position consists from the real position of the *realFMU* simulation (posReal) and the position data of the network (posNet). The mean squared error for the velocity consists of the real velocity of the *realFMU* simulation (velReal) and the velocity data of the network (velNet).
$$ loss = \frac{1}{2} \Bigl[ \frac{1}{n} \sum\limits_{i=0}^n (posReal[i] - posNet[i])^2 + \frac{1}{n} \sum\limits_{i=0}^n (velReal[i] - velNet[i])^2 \Bigr]$$


```julia
# loss function for training
function lossSum()
    global x₀
    solution = neuralFMU(x₀)

    posNet, velNet = extractPosVel(solution)

    (Flux.Losses.mse(posReal, posNet) + Flux.Losses.mse(velReal, velNet)) / 2.0
end
```




    lossSum (generic function with 1 method)



#### Callback

To output the loss in certain time intervals, a callback is implemented as a function in the following. Here a counter is incremented, every fiftieth pass the loss function is called and the average error is printed out. Also, the parameters for the velocity in the first layer are kept to a fixed value.


```julia
# callback function for training
global counter = 0
function callb()
    global counter, paramsNet
    counter += 1

    # freeze first layer parameters (2,4,6) for velocity -> (static) direct feed trough for velocity
    # parameters for position (1,3,5) are learned
    paramsNet[1][2] = 0.0
    paramsNet[1][4] = 1.0
    paramsNet[1][6] = 0.0

    if counter % 50 == 1
        avgLoss = lossSum()
        @info "  Loss [$counter]: $(round(avgLoss, digits=5))
        Avg displacement in data: $(round(sqrt(avgLoss), digits=5))
        Weight/Scale: $(paramsNet[1][1])   Bias/Offset: $(paramsNet[1][5])"
    end
end
```




    callb (generic function with 1 method)



#### Functions for plotting

In this section some important functions for plotting are defined. The function `generate_figure()` creates a new figure object and sets some attributes.


```julia
function generate_figure(title, xLabel, yLabel, xlim="auto")
    Plots.plot(
        title=title, xlabel=xLabel, ylabel=yLabel, linewidth=2,
        xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12, legend=:topright, xlim=xlim)
end
```




    generate_figure (generic function with 2 methods)



In the following function, the data of the *realFMU*, *simpleFMU* and *neuralFMU* are summarized and displayed in a graph.


```julia
function plot_results(title, xLabel, yLabel, interval, realData, simpleData, neuralData)
    linestyles = [:dot, :solid]
    
    fig = generate_figure(title, xLabel, yLabel)
    Plots.plot!(fig, interval, simpleData, label="SimpleFMU", linewidth=2)
    Plots.plot!(fig, interval, realData, label="Reference", linewidth=2)
    for i in 1:length(neuralData)
        Plots.plot!(fig, neuralData[i][1], neuralData[i][2], label="NeuralFMU ($(i*2500))", 
                    linewidth=2, linestyle=linestyles[i], linecolor=:green)
    end
    Plots.display(fig)
end
```




    plot_results (generic function with 1 method)



This is the superordinate function, which at the beginning extracts the position and velocity from the simulation data (`realSimData`, `realSimDataMod`, `simpleSimData`,..., `solutionAfterMod`). Four graphs are then generated, each comparing the corresponding data from the *realFMU*, *simpleFMU*, and *neuralFMU*. The comparison is made with the simulation data from the simulation with the default and modified initial states. According to the data, the designation of the title and the naming of the axes is adapted.


```julia
function plot_all_results(realSimData, realSimDataMod, simpleSimData, 
        simpleSimDataMod, solutionAfter, solutionAfterMod)    
    # collect all data
    posReal, velReal = extractPosVel(realSimData)
    posRealMod, velRealMod = extractPosVel(realSimDataMod)
    posSimple, velSimple = extractPosVel(simpleSimData)
    posSimpleMod, velSimpleMod = extractPosVel(simpleSimDataMod)
    
    run = length(solutionAfter)
    
    posNeural, velNeural = [], []
    posNeuralMod, velNeuralMod = [], []
    for i in 1:run
        dataNeural = extractPosVel(solutionAfter[i])
        time = fmi2GetSolutionTime(solutionAfter[i])

        push!(posNeural, (time, dataNeural[1]))
        push!(velNeural, (time, dataNeural[2]))
        
        dataNeuralMod = extractPosVel(solutionAfterMod[i])
        time = fmi2GetSolutionTime(solutionAfterMod[i])
        push!(posNeuralMod, (time, dataNeuralMod[1]))
        push!(velNeuralMod, (time, dataNeuralMod[2]))
    end
         
    # plot results s (default initial states)
    xLabel="t [s]"
    yLabel="mass position [m]"
    title = "Default: Mass position after Run: $(run)"
    plot_results(title, xLabel, yLabel, tSave, posReal, posSimple, posNeural)

    # plot results s (modified initial states)
    title = "Modified: Mass position after Run: $(run)"
    plot_results(title, xLabel, yLabel, tSave, posRealMod, posSimpleMod, posNeuralMod)

    # plot results v (default initial states)
    yLabel="mass velocity [m/s]"
    title = "Default: Mass velocity after Run: $(run)"
    plot_results(title, xLabel, yLabel, tSave, velReal, velSimple, velNeural)

    # plot results v (modified initial states)    
    title = "Modified: Mass velocity after Run: $(run)"
    plot_results(title, xLabel, yLabel, tSave, velRealMod, velSimpleMod, velNeuralMod)
end
```




    plot_all_results (generic function with 1 method)



The function `plot_friction_model()` compares the friction model of the *realFMU*, *simpleFMU* and *neuralFMU*. For this, the velocity and force from the simulation data of the *realFMU* is needed. The force data is calculated with the extracted last layer of the *neuralFMU* to the real velocity in line 9 by iterating over the vector `velReal`. In the next rows, the velocity and force data (if available) for each of the three FMUs are combined into a matrix. The first row of the matrix corresponds to the later x-axis and here the velocity is plotted. The second row corresponds to the y-axis and here the force is plotted. This matrix is sorted and plotted by the first entries (velocity) with the function `sortperm()`. The graph with at least three graphs is plotted in line 33. As output this function has the forces of the *neuralFMU*.


```julia
function plot_friction_model(realSimData, netBottom, forces)    
    linestyles = [:dot, :solid]
    
    velReal = fmi2GetSolutionValue(realSimData, "mass.v")
    forceReal = fmi2GetSolutionValue(realSimData, "mass.f")

    push!(forces, zeros(length(velReal)))
    for i in 1:length(velReal)
        forces[end][i] = -netBottom([velReal[i], 0.0])[2]
    end

    run = length(forces) 
    
    fig = generate_figure("Friction model $(run)", "v [m/s]", "friction force [N]", (-1.25, 1.25))

    fricSimple = hcat(velReal, zeros(length(velReal)))
    fricSimple[sortperm(fricSimple[:, 1]), :]
    Plots.plot!(fig, fricSimple[:,1], fricSimple[:,2], label="SimpleFMU", linewidth=2)

    fricReal = hcat(velReal, forceReal)
    fricReal[sortperm(fricReal[:, 1]), :]
    Plots.plot!(fig, fricReal[:,1], fricReal[:,2], label="reference", linewidth=2)

    for i in 1:run
        fricNeural = hcat(velReal, forces[i])
        fricNeural[sortperm(fricNeural[:, 1]), :]
        Plots.plot!(fig, fricNeural[:,1], fricNeural[:,2], label="NeuralFMU ($(i*2500))", 
                    linewidth=2, linestyle=linestyles[i], linecolor=:green)
        @info "Friction model $i mse: $(Flux.Losses.mse(fricNeural[:,2], fricReal[:,2]))"
    end
    flush(stderr)

    Plots.display(fig)
    
    return forces   
end
```




    plot_friction_model (generic function with 1 method)



The following function is used to display the different displacement modells of the *realFMU*, *simpleFMU* and *neuralFMU*. The displacement of the *realFMU* and *simpleFMU* is very trivial and is only a constant. The position data of the *realFMU* is needed to calculate the displacement. The displacement for the *neuralFMU* is calculated using the first extracted layer of the neural network, subtracting the real position and the displacement of the *simpleFMU*. Also in this function, the graphs of the three FMUs are compared in a plot.


```julia
function plot_displacement_model(realSimData, netTop, displacements, tSave, displacement)
    linestyles = [:dot, :solid]
    
    posReal = fmi2GetSolutionValue(realSimData, "mass.s")
    
    push!(displacements, zeros(length(posReal)))
    for i in 1:length(posReal)
        displacements[end][i] = netTop([posReal[i], 0.0])[1] - posReal[i] - displacement
    end

    run = length(displacements)
    fig = generate_figure("Displacement model $(run)", "t [s]", "displacement [m]")
    Plots.plot!(fig, [tSave[1], tSave[end]], [displacement, displacement], label="simpleFMU", linewidth=2)
    Plots.plot!(fig, [tSave[1], tSave[end]], [0.0, 0.0], label="reference", linewidth=2)
    for i in 1:run
        Plots.plot!(fig, tSave, displacements[i], label="NeuralFMU ($(i*2500))", 
                    linewidth=2, linestyle=linestyles[i], linecolor=:green)
    end

    Plots.display(fig)
    
    return displacements
end
```




    plot_displacement_model (generic function with 1 method)



#### Structure of the NeuralFMU

In the following, the topology of the NeuralFMU is constructed. It consists of a dense layer that has exactly as many inputs and outputs as the model has states `numStates` (and therefore state derivatives). It also sets the initial weights and offsets for the first dense layer, as well as the activation function, which consists of the identity. An input layer follows, which then leads into the *simpleFMU* model. The ME-FMU computes the state derivatives for a given system state. Following the *simpleFMU* is a dense layer that has `numStates` states. The output of this layer consists of 8 output nodes and a *identity* activation function. The next layer has 8 input and output nodes with a *tanh* activation function. The last layer is again a dense layer with 8 input nodes and the number of states as outputs. Here, it is important that no *tanh*-activation function follows, because otherwise the pendulums state values would be limited to the interval $[-1;1]$.


```julia
# NeuralFMU setup
numStates = fmiGetNumberOfStates(simpleFMU)

# diagonal matrix 
initW = zeros(numStates, numStates)
for i in 1:numStates
    initW[i,i] = 1
end

net = Chain(Dense(initW, zeros(numStates),  identity),
            inputs -> fmiEvaluateME(simpleFMU, inputs),
            Dense(numStates, 8, identity),
            Dense(8, 8, tanh),
            Dense(8, numStates))
```




    Chain(
      Dense(2 => 2),                        [90m# 6 parameters[39m
      var"#1#2"(),
      Dense(2 => 8),                        [90m# 24 parameters[39m
      Dense(8 => 8, tanh),                  [90m# 72 parameters[39m
      Dense(8 => 2),                        [90m# 18 parameters[39m
    ) [90m                  # Total: 8 arrays, [39m120 parameters, 1016 bytes.



#### Definition of the NeuralFMU

The instantiation of the ME-NeuralFMU is done as a one-liner. The FMU (*simpleFMU*), the structure of the network `net`, start `tStart` and end time `tStop`, the numerical solver `Tsit5()` and the time steps `tSave` for saving are specified.


```julia
neuralFMU = ME_NeuralFMU(simpleFMU, net, (tStart, tStop), Tsit5(); saveat=tSave);
```

#### Plot before training

Here the state trajectory of the *simpleFMU* is recorded. Doesn't really look like a pendulum yet, but the system is random initialized by default. In the plots later on, the effect of learning can be seen.


```julia
solutionBefore = neuralFMU(x₀)
fmiPlot(solutionBefore)
```




    
![svg](modelica_conference_2021_files/modelica_conference_2021_45_0.svg)
    



#### Training of the NeuralFMU

For the training of the NeuralFMU the parameters are extracted. All parameters of the first layer are set to the absolute value.


```julia
# train
paramsNet = Flux.params(neuralFMU)

for i in 1:length(paramsNet[1])
    if paramsNet[1][i] < 0.0 
        paramsNet[1][i] = -paramsNet[1][i]
    end
end
```

The well-known ADAM optimizer for minimizing the gradient descent is used as further passing parameters. Additionally, the previously defined loss and callback function as well as a one for the number of epochs are passed. Only one epoch is trained so that the NeuralFMU is precompiled.


```julia
optim = ADAM()
Flux.train!(lossSum, paramsNet, Iterators.repeated((), 1), optim; cb=callb) 
```

    ┌ Info:   Loss [1]: 0.38766
    │         Avg displacement in data: 0.62262
    │         Weight/Scale: 1.0009999999789518   Bias/Offset: 0.0009999999760423974
    └ @ Main In[13]:15


Some vectors for collecting data are initialized and the number of runs, epochs and iterations are set.


```julia
solutionAfter = []
solutionAfterMod = []
forces = []
displacements = []

numRuns = 2
numEpochs= 5
numIterations = 500;
```

#### Training loop

The code section shown here represents the training loop. The loop is structured so that it has `numRuns` runs, where each run has `numEpochs` epochs, and the training is performed at each epoch with `numIterations` iterations. In lines 9 and 10, the data for the *neuralFMU* for the default and modified initial states are appended to the corresponding vectors. The plots for the opposition of position and velocity is done in line 13 by calling the function `plot_all_results`. In the following lines the last layers are extracted from the *neuralFMU* and formed into an independent network `netBottom`. The parameters for the `netBottom` network come from the original architecture and are shared. In line 20, the new network is used to represent the friction model in a graph. An analogous construction of the next part of the training loop, where here the first layer is taken from the *neuralFMU* and converted to its own network `netTop`. This network is used to record the displacement model. The different graphs are generated for each run and can thus be compared. 


```julia
for run in 1:numRuns
    @time for epoch in 1:numEpochs
        @info "Run: $(run)/$(numRuns)  Epoch: $(epoch)/$(numEpochs)"
        Flux.train!(lossSum, paramsNet, Iterators.repeated((), numIterations), optim; cb=callb)
    end
    flush(stderr)
    flush(stdout)
    
    push!(solutionAfter, neuralFMU(x₀))
    push!(solutionAfterMod, neuralFMU(xMod₀))

    # generate all plots for the position and velocity
    plot_all_results(realSimData, realSimDataMod, simpleSimData, simpleSimDataMod, solutionAfter, solutionAfterMod)
    
    # friction model extraction
    layersBottom = neuralFMU.neuralODE.model.layers[3:5]
    netBottom = Chain(layersBottom...)
    transferFlatParams!(netBottom, paramsNet, 7)
    
    forces = plot_friction_model(realSimData, netBottom, forces) 
    
    # displacement model extraction
    layersTop = neuralFMU.neuralODE.model.layers[1:1]
    netTop = Chain(layersTop...)
    transferFlatParams!(netTop, paramsNet, 1)

    displacements = plot_displacement_model(realSimData, netTop, displacements, tSave, displacement)
end
```

    ┌ Info: Run: 1/2  Epoch: 1/5
    └ @ Main In[25]:3
    ┌ Info:   Loss [51]: 0.27987
    │         Avg displacement in data: 0.52903
    │         Weight/Scale: 1.0227796972484995   Bias/Offset: 0.02359101781398293
    └ @ Main In[13]:15
    ┌ Info:   Loss [101]: 0.23462
    │         Avg displacement in data: 0.48438
    │         Weight/Scale: 1.007235304135815   Bias/Offset: 0.012056601842674715
    └ @ Main In[13]:15
    ┌ Info:   Loss [151]: 0.07568
    │         Avg displacement in data: 0.27511
    │         Weight/Scale: 1.025558013690243   Bias/Offset: 0.054126679913574706
    └ @ Main In[13]:15
    ┌ Info:   Loss [201]: 0.03432
    │         Avg displacement in data: 0.18527
    │         Weight/Scale: 1.0514125925850004   Bias/Offset: 0.0836581766086249
    └ @ Main In[13]:15
    ┌ Info:   Loss [251]: 0.03042
    │         Avg displacement in data: 0.1744
    │         Weight/Scale: 1.0527700657960908   Bias/Offset: 0.08237075469577392
    └ @ Main In[13]:15
    ┌ Info:   Loss [301]: 0.02736
    │         Avg displacement in data: 0.1654
    │         Weight/Scale: 1.052739942802182   Bias/Offset: 0.0800787153236984
    └ @ Main In[13]:15
    ┌ Info:   Loss [351]: 0.02485
    │         Avg displacement in data: 0.15763
    │         Weight/Scale: 1.0518631427268665   Bias/Offset: 0.07737129021651924
    └ @ Main In[13]:15
    ┌ Info:   Loss [401]: 0.02285
    │         Avg displacement in data: 0.15116
    │         Weight/Scale: 1.0504819371354388   Bias/Offset: 0.07499401573025632
    └ @ Main In[13]:15
    ┌ Info:   Loss [451]: 0.02139
    │         Avg displacement in data: 0.14626
    │         Weight/Scale: 1.0485161632628934   Bias/Offset: 0.07266634553128398
    └ @ Main In[13]:15
    ┌ Info:   Loss [501]: 0.0202
    │         Avg displacement in data: 0.14212
    │         Weight/Scale: 1.0465161927062194   Bias/Offset: 0.07103174526720096
    └ @ Main In[13]:15
    ┌ Info: Run: 1/2  Epoch: 2/5
    └ @ Main In[25]:3
    ┌ Info:   Loss [551]: 0.01926
    │         Avg displacement in data: 0.13878
    │         Weight/Scale: 1.044526487305734   Bias/Offset: 0.0696085156424645
    └ @ Main In[13]:15
    ┌ Info:   Loss [601]: 0.01837
    │         Avg displacement in data: 0.13555
    │         Weight/Scale: 1.0423604509663782   Bias/Offset: 0.06830160565203962
    └ @ Main In[13]:15
    ┌ Info:   Loss [651]: 0.01757
    │         Avg displacement in data: 0.13254
    │         Weight/Scale: 1.0401544042429705   Bias/Offset: 0.06742281384546482
    └ @ Main In[13]:15
    ┌ Info:   Loss [701]: 0.01687
    │         Avg displacement in data: 0.12989
    │         Weight/Scale: 1.0378657538835463   Bias/Offset: 0.06655857932091809
    └ @ Main In[13]:15
    ┌ Info:   Loss [751]: 0.01605
    │         Avg displacement in data: 0.12669
    │         Weight/Scale: 1.0356939934301541   Bias/Offset: 0.06596511697735462
    └ @ Main In[13]:15
    ┌ Info:   Loss [801]: 0.01529
    │         Avg displacement in data: 0.12365
    │         Weight/Scale: 1.0334524191183936   Bias/Offset: 0.06555964738908121
    └ @ Main In[13]:15
    ┌ Info:   Loss [851]: 0.01444
    │         Avg displacement in data: 0.12019
    │         Weight/Scale: 1.0311602846871473   Bias/Offset: 0.06519152717886686
    └ @ Main In[13]:15
    ┌ Info:   Loss [901]: 0.01353
    │         Avg displacement in data: 0.11632
    │         Weight/Scale: 1.028802105007811   Bias/Offset: 0.06472347619981693
    └ @ Main In[13]:15
    ┌ Info:   Loss [951]: 0.0126
    │         Avg displacement in data: 0.11227
    │         Weight/Scale: 1.0264555403375735   Bias/Offset: 0.06404485810878471
    └ @ Main In[13]:15
    ┌ Info:   Loss [1001]: 0.01171
    │         Avg displacement in data: 0.10823
    │         Weight/Scale: 1.0244338972990028   Bias/Offset: 0.06356782644489042
    └ @ Main In[13]:15
    ┌ Info: Run: 1/2  Epoch: 3/5
    └ @ Main In[25]:3
    ┌ Info:   Loss [1051]: 0.01087
    │         Avg displacement in data: 0.10425
    │         Weight/Scale: 1.0226052761883384   Bias/Offset: 0.0631451807018007
    └ @ Main In[13]:15
    ┌ Info:   Loss [1101]: 0.01008
    │         Avg displacement in data: 0.10039
    │         Weight/Scale: 1.0211340738544854   Bias/Offset: 0.06302604534753369
    └ @ Main In[13]:15
    ┌ Info:   Loss [1151]: 0.00935
    │         Avg displacement in data: 0.0967
    │         Weight/Scale: 1.0198029416973389   Bias/Offset: 0.06295024095606516
    └ @ Main In[13]:15
    ┌ Info:   Loss [1201]: 0.00867
    │         Avg displacement in data: 0.0931
    │         Weight/Scale: 1.0185473991787677   Bias/Offset: 0.06288464728495828
    └ @ Main In[13]:15
    ┌ Info:   Loss [1251]: 0.00801
    │         Avg displacement in data: 0.08952
    │         Weight/Scale: 1.017286168241178   Bias/Offset: 0.06276506687507463
    └ @ Main In[13]:15
    ┌ Info:   Loss [1301]: 0.00729
    │         Avg displacement in data: 0.0854
    │         Weight/Scale: 1.0159057817482948   Bias/Offset: 0.062432243830558926
    └ @ Main In[13]:15
    ┌ Info:   Loss [1351]: 0.00655
    │         Avg displacement in data: 0.08093
    │         Weight/Scale: 1.0146956856856428   Bias/Offset: 0.062136078392318314
    └ @ Main In[13]:15
    ┌ Info:   Loss [1401]: 0.00588
    │         Avg displacement in data: 0.07665
    │         Weight/Scale: 1.0134994843490717   Bias/Offset: 0.0614479564650401
    └ @ Main In[13]:15
    ┌ Info:   Loss [1451]: 0.00538
    │         Avg displacement in data: 0.07336
    │         Weight/Scale: 1.0126014399836643   Bias/Offset: 0.060789236747274565
    └ @ Main In[13]:15
    ┌ Info:   Loss [1501]: 0.005
    │         Avg displacement in data: 0.07071
    │         Weight/Scale: 1.011897256968808   Bias/Offset: 0.0602251854840801
    └ @ Main In[13]:15
    ┌ Info: Run: 1/2  Epoch: 4/5
    └ @ Main In[25]:3
    ┌ Info:   Loss [1551]: 0.00468
    │         Avg displacement in data: 0.06838
    │         Weight/Scale: 1.0113079761138857   Bias/Offset: 0.05976182356133882
    └ @ Main In[13]:15
    ┌ Info:   Loss [1601]: 0.00439
    │         Avg displacement in data: 0.06626
    │         Weight/Scale: 1.0107772861145496   Bias/Offset: 0.05934811678367015
    └ @ Main In[13]:15
    ┌ Info:   Loss [1651]: 0.00414
    │         Avg displacement in data: 0.06434
    │         Weight/Scale: 1.0102864663334739   Bias/Offset: 0.0589636441017932
    └ @ Main In[13]:15
    ┌ Info:   Loss [1701]: 0.0039
    │         Avg displacement in data: 0.06249
    │         Weight/Scale: 1.0098281047278808   Bias/Offset: 0.05860012703210355
    └ @ Main In[13]:15
    ┌ Info:   Loss [1751]: 0.00371
    │         Avg displacement in data: 0.06088
    │         Weight/Scale: 1.0094199153396044   Bias/Offset: 0.05828631837620575
    └ @ Main In[13]:15
    ┌ Info:   Loss [1801]: 0.00352
    │         Avg displacement in data: 0.05931
    │         Weight/Scale: 1.0090414444264393   Bias/Offset: 0.05799325827066244
    └ @ Main In[13]:15
    ┌ Info:   Loss [1851]: 0.00335
    │         Avg displacement in data: 0.05787
    │         Weight/Scale: 1.00865356350199   Bias/Offset: 0.05766959382290175
    └ @ Main In[13]:15
    ┌ Info:   Loss [1901]: 0.0032
    │         Avg displacement in data: 0.05653
    │         Weight/Scale: 1.008273845259956   Bias/Offset: 0.057339985902657396
    └ @ Main In[13]:15
    ┌ Info:   Loss [1951]: 0.00306
    │         Avg displacement in data: 0.05529
    │         Weight/Scale: 1.0079173630571645   Bias/Offset: 0.057026745844088136
    └ @ Main In[13]:15
    ┌ Info:   Loss [2001]: 0.00293
    │         Avg displacement in data: 0.05414
    │         Weight/Scale: 1.0075836627207844   Bias/Offset: 0.05673195485611518
    └ @ Main In[13]:15
    ┌ Info: Run: 1/2  Epoch: 5/5
    └ @ Main In[25]:3
    ┌ Info:   Loss [2051]: 0.00282
    │         Avg displacement in data: 0.05306
    │         Weight/Scale: 1.0072697334383847   Bias/Offset: 0.05645350581842648
    └ @ Main In[13]:15
    ┌ Info:   Loss [2101]: 0.00271
    │         Avg displacement in data: 0.05205
    │         Weight/Scale: 1.0069730034407365   Bias/Offset: 0.05618931211150019
    └ @ Main In[13]:15
    ┌ Info:   Loss [2151]: 0.00261
    │         Avg displacement in data: 0.05109
    │         Weight/Scale: 1.0066920282308016   Bias/Offset: 0.0559385401371683
    └ @ Main In[13]:15
    ┌ Info:   Loss [2201]: 0.00252
    │         Avg displacement in data: 0.05019
    │         Weight/Scale: 1.0064261392097056   Bias/Offset: 0.05570140397037317
    └ @ Main In[13]:15
    ┌ Info:   Loss [2251]: 0.00243
    │         Avg displacement in data: 0.04934
    │         Weight/Scale: 1.0061738368294062   Bias/Offset: 0.055476572151725925
    └ @ Main In[13]:15
    ┌ Info:   Loss [2301]: 0.00236
    │         Avg displacement in data: 0.04853
    │         Weight/Scale: 1.00593327091279   Bias/Offset: 0.055261663260595055
    └ @ Main In[13]:15
    ┌ Info:   Loss [2351]: 0.00228
    │         Avg displacement in data: 0.04776
    │         Weight/Scale: 1.005702899132193   Bias/Offset: 0.05505465426625899
    └ @ Main In[13]:15
    ┌ Info:   Loss [2401]: 0.00221
    │         Avg displacement in data: 0.04703
    │         Weight/Scale: 1.0054818817969449   Bias/Offset: 0.05485471756699228
    └ @ Main In[13]:15
    ┌ Info:   Loss [2451]: 0.00215
    │         Avg displacement in data: 0.04633
    │         Weight/Scale: 1.0052696914341415   Bias/Offset: 0.05466166484597469
    └ @ Main In[13]:15
    ┌ Info:   Loss [2501]: 0.00209
    │         Avg displacement in data: 0.04566
    │         Weight/Scale: 1.0050656974011385   Bias/Offset: 0.05447527099558027
    └ @ Main In[13]:15


    125.281003 seconds (307.64 M allocations: 121.448 GiB, 11.81% gc time)



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_2.svg)
    


    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_4.svg)
    


    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_6.svg)
    


    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_8.svg)
    


    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Info: Friction model 1 mse: 0.5841838178967346
    └ @ Main In[17]:29



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_10.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_11.svg)
    


    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Info: Run: 2/2  Epoch: 1/5
    └ @ Main In[25]:3
    ┌ Info:   Loss [2551]: 0.00203
    │         Avg displacement in data: 0.04502
    │         Weight/Scale: 1.0048691798921354   Bias/Offset: 0.054295074682339774
    └ @ Main In[13]:15
    ┌ Info:   Loss [2601]: 0.00197
    │         Avg displacement in data: 0.04441
    │         Weight/Scale: 1.0046797928811213   Bias/Offset: 0.054120864493847884
    └ @ Main In[13]:15
    ┌ Info:   Loss [2651]: 0.00192
    │         Avg displacement in data: 0.04382
    │         Weight/Scale: 1.0044971219024164   Bias/Offset: 0.05395227390510312
    └ @ Main In[13]:15
    ┌ Info:   Loss [2701]: 0.00187
    │         Avg displacement in data: 0.04325
    │         Weight/Scale: 1.0043205142799836   Bias/Offset: 0.0537885752852632
    └ @ Main In[13]:15
    ┌ Info:   Loss [2751]: 0.00182
    │         Avg displacement in data: 0.0427
    │         Weight/Scale: 1.0041495131775346   Bias/Offset: 0.05362928288258972
    └ @ Main In[13]:15
    ┌ Info:   Loss [2801]: 0.00178
    │         Avg displacement in data: 0.04217
    │         Weight/Scale: 1.0039836109760338   Bias/Offset: 0.05347366108733261
    └ @ Main In[13]:15
    ┌ Info:   Loss [2851]: 0.00174
    │         Avg displacement in data: 0.04166
    │         Weight/Scale: 1.0038223517091698   Bias/Offset: 0.05332106138776155
    └ @ Main In[13]:15
    ┌ Info:   Loss [2901]: 0.0017
    │         Avg displacement in data: 0.04117
    │         Weight/Scale: 1.0036641595719626   Bias/Offset: 0.05316979060838563
    └ @ Main In[13]:15
    ┌ Info:   Loss [2951]: 0.00166
    │         Avg displacement in data: 0.0407
    │         Weight/Scale: 1.0034824771593116   Bias/Offset: 0.053023815200695865
    └ @ Main In[13]:15
    ┌ Info:   Loss [3001]: 0.00162
    │         Avg displacement in data: 0.04024
    │         Weight/Scale: 1.003302287232279   Bias/Offset: 0.05284652737193568
    └ @ Main In[13]:15
    ┌ Info: Run: 2/2  Epoch: 2/5
    └ @ Main In[25]:3
    ┌ Info:   Loss [3051]: 0.00158
    │         Avg displacement in data: 0.0398
    │         Weight/Scale: 1.0031327356622615   Bias/Offset: 0.05267427236442638
    └ @ Main In[13]:15
    ┌ Info:   Loss [3101]: 0.00156
    │         Avg displacement in data: 0.03953
    │         Weight/Scale: 1.0029500684407646   Bias/Offset: 0.05248895808613692
    └ @ Main In[13]:15
    ┌ Info:   Loss [3151]: 0.00152
    │         Avg displacement in data: 0.03897
    │         Weight/Scale: 1.0027515706381962   Bias/Offset: 0.05231557617707483
    └ @ Main In[13]:15
    ┌ Info:   Loss [3201]: 0.00149
    │         Avg displacement in data: 0.03857
    │         Weight/Scale: 1.0025671518586756   Bias/Offset: 0.05211843562441854
    └ @ Main In[13]:15
    ┌ Info:   Loss [3251]: 0.00146
    │         Avg displacement in data: 0.03818
    │         Weight/Scale: 1.0023819730626184   Bias/Offset: 0.05191540591786916
    └ @ Main In[13]:15
    ┌ Info:   Loss [3301]: 0.00144
    │         Avg displacement in data: 0.03791
    │         Weight/Scale: 1.0021632889377226   Bias/Offset: 0.05171860500099298
    └ @ Main In[13]:15
    ┌ Info:   Loss [3351]: 0.0014
    │         Avg displacement in data: 0.03744
    │         Weight/Scale: 1.0019411794517727   Bias/Offset: 0.05147520346434769
    └ @ Main In[13]:15
    ┌ Info:   Loss [3401]: 0.00138
    │         Avg displacement in data: 0.03709
    │         Weight/Scale: 1.0017329315033845   Bias/Offset: 0.0512361956105694
    └ @ Main In[13]:15
    ┌ Info:   Loss [3451]: 0.00135
    │         Avg displacement in data: 0.03674
    │         Weight/Scale: 1.0015212202792987   Bias/Offset: 0.050989154848838064
    └ @ Main In[13]:15
    ┌ Info:   Loss [3501]: 0.00133
    │         Avg displacement in data: 0.03642
    │         Weight/Scale: 1.0012557067126089   Bias/Offset: 0.0507363501769044
    └ @ Main In[13]:15
    ┌ Info: Run: 2/2  Epoch: 3/5
    └ @ Main In[25]:3
    ┌ Info:   Loss [3551]: 0.0013
    │         Avg displacement in data: 0.03608
    │         Weight/Scale: 1.00101407962594   Bias/Offset: 0.050454277227812774
    └ @ Main In[13]:15
    ┌ Info:   Loss [3601]: 0.00128
    │         Avg displacement in data: 0.03576
    │         Weight/Scale: 1.0007733249334747   Bias/Offset: 0.050161991984925096
    └ @ Main In[13]:15
    ┌ Info:   Loss [3651]: 0.00126
    │         Avg displacement in data: 0.03545
    │         Weight/Scale: 1.0005272517056891   Bias/Offset: 0.049859255147195664
    └ @ Main In[13]:15
    ┌ Info:   Loss [3701]: 0.00124
    │         Avg displacement in data: 0.0352
    │         Weight/Scale: 1.0002271224454813   Bias/Offset: 0.04955690861694724
    └ @ Main In[13]:15
    ┌ Info:   Loss [3751]: 0.00121
    │         Avg displacement in data: 0.03485
    │         Weight/Scale: 0.9999339221810825   Bias/Offset: 0.04920328312338153
    └ @ Main In[13]:15
    ┌ Info:   Loss [3801]: 0.00119
    │         Avg displacement in data: 0.03456
    │         Weight/Scale: 0.9996551008765587   Bias/Offset: 0.04884717631531261
    └ @ Main In[13]:15
    ┌ Info:   Loss [3851]: 0.00117
    │         Avg displacement in data: 0.03428
    │         Weight/Scale: 0.9993674736621564   Bias/Offset: 0.04847594692147291
    └ @ Main In[13]:15
    ┌ Info:   Loss [3901]: 0.00116
    │         Avg displacement in data: 0.03405
    │         Weight/Scale: 0.9990035565890562   Bias/Offset: 0.048085493708647266
    └ @ Main In[13]:15
    ┌ Info:   Loss [3951]: 0.00114
    │         Avg displacement in data: 0.03372
    │         Weight/Scale: 0.9986868123334718   Bias/Offset: 0.04767576133969908
    └ @ Main In[13]:15
    ┌ Info:   Loss [4001]: 0.00112
    │         Avg displacement in data: 0.03345
    │         Weight/Scale: 0.9983607611565413   Bias/Offset: 0.047241400912823174
    └ @ Main In[13]:15
    ┌ Info: Run: 2/2  Epoch: 4/5
    └ @ Main In[25]:3
    ┌ Info:   Loss [4051]: 0.00111
    │         Avg displacement in data: 0.03335
    │         Weight/Scale: 0.9980437088605549   Bias/Offset: 0.046803904120799336
    └ @ Main In[13]:15
    ┌ Info:   Loss [4101]: 0.00108
    │         Avg displacement in data: 0.03292
    │         Weight/Scale: 0.9976115569702646   Bias/Offset: 0.046317907987247844
    └ @ Main In[13]:15
    ┌ Info:   Loss [4151]: 0.00107
    │         Avg displacement in data: 0.03266
    │         Weight/Scale: 0.9972363139527056   Bias/Offset: 0.04580947813192842
    └ @ Main In[13]:15
    ┌ Info:   Loss [4201]: 0.00105
    │         Avg displacement in data: 0.0324
    │         Weight/Scale: 0.9968563641088443   Bias/Offset: 0.045282006620128004
    └ @ Main In[13]:15
    ┌ Info:   Loss [4251]: 0.00105
    │         Avg displacement in data: 0.03236
    │         Weight/Scale: 0.9964605428921461   Bias/Offset: 0.044775633694189604
    └ @ Main In[13]:15
    ┌ Info:   Loss [4301]: 0.00102
    │         Avg displacement in data: 0.0319
    │         Weight/Scale: 0.9959804996736025   Bias/Offset: 0.04415144225670856
    └ @ Main In[13]:15
    ┌ Info:   Loss [4351]: 0.001
    │         Avg displacement in data: 0.03165
    │         Weight/Scale: 0.9955469063544566   Bias/Offset: 0.04353634708671823
    └ @ Main In[13]:15
    ┌ Info:   Loss [4401]: 0.00099
    │         Avg displacement in data: 0.0314
    │         Weight/Scale: 0.9950986697472873   Bias/Offset: 0.042893732718874505
    └ @ Main In[13]:15
    ┌ Info:   Loss [4451]: 0.00097
    │         Avg displacement in data: 0.03122
    │         Weight/Scale: 0.994563957898894   Bias/Offset: 0.04223954464266206
    └ @ Main In[13]:15
    ┌ Info:   Loss [4501]: 0.00096
    │         Avg displacement in data: 0.03091
    │         Weight/Scale: 0.9940501319202598   Bias/Offset: 0.04151719347053684
    └ @ Main In[13]:15
    ┌ Info: Run: 2/2  Epoch: 5/5
    └ @ Main In[25]:3
    ┌ Info:   Loss [4551]: 0.00094
    │         Avg displacement in data: 0.03067
    │         Weight/Scale: 0.9935450273992231   Bias/Offset: 0.04077214664070263
    └ @ Main In[13]:15
    ┌ Info:   Loss [4601]: 0.00093
    │         Avg displacement in data: 0.03043
    │         Weight/Scale: 0.993019980577346   Bias/Offset: 0.03999220242638018
    └ @ Main In[13]:15
    ┌ Info:   Loss [4651]: 0.00096
    │         Avg displacement in data: 0.03094
    │         Weight/Scale: 0.9924879797604587   Bias/Offset: 0.039197530860889805
    └ @ Main In[13]:15
    ┌ Info:   Loss [4701]: 0.0009
    │         Avg displacement in data: 0.02995
    │         Weight/Scale: 0.9918131259637384   Bias/Offset: 0.03832419443064244
    └ @ Main In[13]:15
    ┌ Info:   Loss [4751]: 0.00088
    │         Avg displacement in data: 0.02971
    │         Weight/Scale: 0.9912200146454442   Bias/Offset: 0.03742704315091618
    └ @ Main In[13]:15
    ┌ Info:   Loss [4801]: 0.00087
    │         Avg displacement in data: 0.02948
    │         Weight/Scale: 0.9906035857563007   Bias/Offset: 0.03648401090788386
    └ @ Main In[13]:15
    ┌ Info:   Loss [4851]: 0.00086
    │         Avg displacement in data: 0.0293
    │         Weight/Scale: 0.9899467513927472   Bias/Offset: 0.035483836954150375
    └ @ Main In[13]:15
    ┌ Info:   Loss [4901]: 0.00084
    │         Avg displacement in data: 0.029
    │         Weight/Scale: 0.9892036550465079   Bias/Offset: 0.034467675192321266
    └ @ Main In[13]:15
    ┌ Info:   Loss [4951]: 0.00083
    │         Avg displacement in data: 0.02876
    │         Weight/Scale: 0.988508173693831   Bias/Offset: 0.03339060594234028
    └ @ Main In[13]:15
    ┌ Info:   Loss [5001]: 0.00081
    │         Avg displacement in data: 0.02853
    │         Weight/Scale: 0.987786770191395   Bias/Offset: 0.032256729968733
    └ @ Main In[13]:15


    116.437996 seconds (284.00 M allocations: 112.776 GiB, 12.02% gc time)



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_14.svg)
    


    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_16.svg)
    


    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_18.svg)
    


    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_20.svg)
    


    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Info: Friction model 1 mse: 0.5841838178967346
    └ @ Main In[17]:29
    ┌ Info: Friction model 2 mse: 0.8024621143969841
    └ @ Main In[17]:29



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_22.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_53_23.svg)
    


    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595
    ┌ Warning: Invalid limits for x axis. Limits should be a symbol, or a two-element tuple or vector of numbers.
    │ xlims = auto
    └ @ Plots /home/runner/.julia/packages/Plots/nbICw/src/axes.jl:595


Finally, the FMU is cleaned-up.


```julia
fmiUnload(simpleFMU)
```

### Summary

Based on the plots, it can be seen that the curves of the *realFMU* and the *neuralFMU* are very close. The *neuralFMU* is able to learn the friction and displacement model.

### Source

[1] Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

