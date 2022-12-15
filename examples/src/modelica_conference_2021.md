# ME-NeuralFMU from the Modelica Conference 2021
Tutorial by Johannes Stoljar, Tobias Thummerer

## License


```julia
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons, Johannes Stoljar
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.
```

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
Besides, this [Jupyter Notebook](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021.ipynb) there is also a [Julia file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021.jl) with the same name, which contains only the code cells. For the documentation there is a [Markdown file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021.md) corresponding to the notebook.  


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

realSimData = fmiSimulate(realFMU, (tStart, tStop); parameters=params, recordValues=vrs, saveat=tSave)
posReal = fmi2GetSolutionValue(realSimData, "mass.s")
velReal = fmi2GetSolutionValue(realSimData, "mass.v")
fmiPlot(realSimData)
```




    
![svg](modelica_conference_2021_files/modelica_conference_2021_12_0.svg)
    



#### Define functions

The structure of the previous code section is used more often in the further sections, so for clarity the previously explained code section for setting the paramters and simulating are combined into one function `simulate()`.


```julia
function simulate(FMU, initStates, x₀, variables, tStart, tStop, tSave)
    params = Dict(zip(initStates, x₀))
    return fmiSimulate(FMU, (tStart, tStop); parameters=params, recordValues=variables, saveat=tSave)
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




    
![svg](modelica_conference_2021_files/modelica_conference_2021_18_0.svg)
    



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




    
![svg](modelica_conference_2021_files/modelica_conference_2021_25_0.svg)
    



#### Modified initial states

The same values for the initial states are used for this simulation as for the simulation from the *realFMU* with the modified initial states.


```julia
xSimpleMod₀ = vcat(xMod₀, displacement)

simpleSimDataMod = simulate(simpleFMU, initStates, xSimpleMod₀, vrs, tStart, tStop, tSave)
fmiPlot(simpleSimDataMod)
```




    
![svg](modelica_conference_2021_files/modelica_conference_2021_27_0.svg)
    



## NeuralFMU

#### Loss function

In order to train our model, a loss function must be implemented. The solver of the NeuralFMU can calculate the gradient of the loss function. The gradient descent is needed to adjust the weights in the neural network so that the sum of the error is reduced and the model becomes more accurate.

The error function in this implementation consists of the mean of the mean squared errors. The first part of the addition is the deviation of the position and the second part is the deviation of the velocity. The mean squared error (mse) for the position consists from the real position of the *realFMU* simulation (posReal) and the position data of the network (posNet). The mean squared error for the velocity consists of the real velocity of the *realFMU* simulation (velReal) and the velocity data of the network (velNet).
$$ loss = \frac{1}{2} \Bigl[ \frac{1}{n} \sum\limits_{i=0}^n (posReal[i] - posNet[i])^2 + \frac{1}{n} \sum\limits_{i=0}^n (velReal[i] - velNet[i])^2 \Bigr]$$


```julia
# loss function for training
function lossSum(p)
    global x₀
    solution = neuralFMU(x₀; p=p)

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
function callb(p)
    global counter
    counter += 1

    # freeze first layer parameters (2,4,6) for velocity -> (static) direct feed trough for velocity
    # parameters for position (1,3,5) are learned
    p[1][2] = 0.0
    p[1][4] = 1.0
    p[1][6] = 0.0

    if counter % 50 == 1
        avgLoss = lossSum(p[1])
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
function generate_figure(title, xLabel, yLabel, xlim=:auto)
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

net = Chain(# Dense(initW, zeros(numStates),  identity),
            Dense(numStates, numStates,  identity),
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
    ) [90m                  # Total: 8 arrays, [39m120 parameters, 992 bytes.



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




    
![svg](modelica_conference_2021_files/modelica_conference_2021_47_0.svg)
    



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
FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), 1), optim; cb=()->callb(paramsNet)) 
```

    ┌ Info:   Loss [1]: 0.64142
    │         Avg displacement in data: 0.80089
    │         Weight/Scale: 0.5550727972915012   Bias/Offset: 0.0009999999900079759
    └ @ Main In[14]:15


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
        FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), numIterations), optim; cb=()->callb(paramsNet))
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
    └ @ Main In[26]:3
    ┌ Info:   Loss [51]: 0.45485
    │         Avg displacement in data: 0.67443
    │         Weight/Scale: 0.6028802404716107   Bias/Offset: 0.04828021006157077
    └ @ Main In[14]:15
    ┌ Info:   Loss [101]: 0.39138
    │         Avg displacement in data: 0.6256
    │         Weight/Scale: 0.6409413544487673   Bias/Offset: 0.08734044707986238
    └ @ Main In[14]:15
    ┌ Info:   Loss [151]: 0.35729
    │         Avg displacement in data: 0.59774
    │         Weight/Scale: 0.6705024458804647   Bias/Offset: 0.11917412473236139
    └ @ Main In[14]:15
    ┌ Info:   Loss [201]: 0.33751
    │         Avg displacement in data: 0.58096
    │         Weight/Scale: 0.6940488159966223   Bias/Offset: 0.14522352322020005
    └ @ Main In[14]:15
    ┌ Info:   Loss [251]: 0.32533
    │         Avg displacement in data: 0.57038
    │         Weight/Scale: 0.7129425125355464   Bias/Offset: 0.16635454654923768
    └ @ Main In[14]:15
    ┌ Info:   Loss [301]: 0.31723
    │         Avg displacement in data: 0.56323
    │         Weight/Scale: 0.7280758621584592   Bias/Offset: 0.18315566333683783
    └ @ Main In[14]:15
    ┌ Info:   Loss [351]: 0.31089
    │         Avg displacement in data: 0.55758
    │         Weight/Scale: 0.7400860511827587   Bias/Offset: 0.19604467290252178
    └ @ Main In[14]:15
    ┌ Info:   Loss [401]: 0.30264
    │         Avg displacement in data: 0.55013
    │         Weight/Scale: 0.7494671100680018   Bias/Offset: 0.20518196529098945
    └ @ Main In[14]:15
    ┌ Info:   Loss [451]: 0.28733
    │         Avg displacement in data: 0.53603
    │         Weight/Scale: 0.7570692033734648   Bias/Offset: 0.21047351091721003
    └ @ Main In[14]:15
    ┌ Info:   Loss [501]: 0.2377
    │         Avg displacement in data: 0.48754
    │         Weight/Scale: 0.7632815153832909   Bias/Offset: 0.2103674948475488
    └ @ Main In[14]:15
    ┌ Info: Run: 1/2  Epoch: 2/5
    └ @ Main In[26]:3
    ┌ Info:   Loss [551]: 0.17519
    │         Avg displacement in data: 0.41855
    │         Weight/Scale: 0.771933610211766   Bias/Offset: 0.21532825586937301
    └ @ Main In[14]:15
    ┌ Info:   Loss [601]: 0.03232
    │         Avg displacement in data: 0.17977
    │         Weight/Scale: 0.7894761928254148   Bias/Offset: 0.24217790496864008
    └ @ Main In[14]:15
    ┌ Info:   Loss [651]: 0.0245
    │         Avg displacement in data: 0.15654
    │         Weight/Scale: 0.7844989979590299   Bias/Offset: 0.23221241549764232
    └ @ Main In[14]:15
    ┌ Info:   Loss [701]: 0.02054
    │         Avg displacement in data: 0.1433
    │         Weight/Scale: 0.7815568611545257   Bias/Offset: 0.2276763035082252
    └ @ Main In[14]:15
    ┌ Info:   Loss [751]: 0.01775
    │         Avg displacement in data: 0.13323
    │         Weight/Scale: 0.7782293876125358   Bias/Offset: 0.22259390253392786
    └ @ Main In[14]:15
    ┌ Info:   Loss [801]: 0.01605
    │         Avg displacement in data: 0.12669
    │         Weight/Scale: 0.7750339742719601   Bias/Offset: 0.2172935230029692
    └ @ Main In[14]:15
    ┌ Info:   Loss [851]: 0.01559
    │         Avg displacement in data: 0.12485
    │         Weight/Scale: 0.7737418532747102   Bias/Offset: 0.21537509834326932
    └ @ Main In[14]:15
    ┌ Info:   Loss [901]: 0.01572
    │         Avg displacement in data: 0.12536
    │         Weight/Scale: 0.7739679038503342   Bias/Offset: 0.2165116055276674
    └ @ Main In[14]:15
    ┌ Info:   Loss [951]: 0.01506
    │         Avg displacement in data: 0.12274
    │         Weight/Scale: 0.7730088206185897   Bias/Offset: 0.2163316203576429
    └ @ Main In[14]:15
    ┌ Info:   Loss [1001]: 0.0148
    │         Avg displacement in data: 0.12166
    │         Weight/Scale: 0.772004119527416   Bias/Offset: 0.21594497741124227
    └ @ Main In[14]:15
    ┌ Info: Run: 1/2  Epoch: 3/5
    └ @ Main In[26]:3
    ┌ Info:   Loss [1051]: 0.01446
    │         Avg displacement in data: 0.12026
    │         Weight/Scale: 0.7683625626309447   Bias/Offset: 0.21082500826760647
    └ @ Main In[14]:15
    ┌ Info:   Loss [1101]: 0.01315
    │         Avg displacement in data: 0.11466
    │         Weight/Scale: 0.7651830111987297   Bias/Offset: 0.20575878480956503
    └ @ Main In[14]:15
    ┌ Info:   Loss [1151]: 0.01397
    │         Avg displacement in data: 0.11819
    │         Weight/Scale: 0.7636138806661388   Bias/Offset: 0.20339378533347718
    └ @ Main In[14]:15
    ┌ Info:   Loss [1201]: 0.01349
    │         Avg displacement in data: 0.11617
    │         Weight/Scale: 0.7618338476729756   Bias/Offset: 0.20075963153895074
    └ @ Main In[14]:15
    ┌ Info:   Loss [1251]: 0.01359
    │         Avg displacement in data: 0.11658
    │         Weight/Scale: 0.759277814668437   Bias/Offset: 0.19630835413406816
    └ @ Main In[14]:15
    ┌ Info:   Loss [1301]: 0.01274
    │         Avg displacement in data: 0.11287
    │         Weight/Scale: 0.7604012648465226   Bias/Offset: 0.19764166833355087
    └ @ Main In[14]:15
    ┌ Info:   Loss [1351]: 0.01212
    │         Avg displacement in data: 0.11009
    │         Weight/Scale: 0.7590017913609226   Bias/Offset: 0.19632666262156273
    └ @ Main In[14]:15
    ┌ Info:   Loss [1401]: 0.0118
    │         Avg displacement in data: 0.10862
    │         Weight/Scale: 0.758122328637297   Bias/Offset: 0.19459760596350487
    └ @ Main In[14]:15
    ┌ Info:   Loss [1451]: 0.01159
    │         Avg displacement in data: 0.10764
    │         Weight/Scale: 0.7592906237938988   Bias/Offset: 0.1970160651375306
    └ @ Main In[14]:15
    ┌ Info:   Loss [1501]: 0.01142
    │         Avg displacement in data: 0.10687
    │         Weight/Scale: 0.7593329615710782   Bias/Offset: 0.19853185115006367
    └ @ Main In[14]:15
    ┌ Info: Run: 1/2  Epoch: 4/5
    └ @ Main In[26]:3
    ┌ Info:   Loss [1551]: 0.01178
    │         Avg displacement in data: 0.10855
    │         Weight/Scale: 0.7588701852604998   Bias/Offset: 0.19960313591551038
    └ @ Main In[14]:15
    ┌ Info:   Loss [1601]: 0.01187
    │         Avg displacement in data: 0.10895
    │         Weight/Scale: 0.7575674288813233   Bias/Offset: 0.19976693878845794
    └ @ Main In[14]:15
    ┌ Info:   Loss [1651]: 0.012
    │         Avg displacement in data: 0.10955
    │         Weight/Scale: 0.756113284039509   Bias/Offset: 0.1996293584896335
    └ @ Main In[14]:15
    ┌ Info:   Loss [1701]: 0.01226
    │         Avg displacement in data: 0.11072
    │         Weight/Scale: 0.754836707403587   Bias/Offset: 0.19944115329323944
    └ @ Main In[14]:15
    ┌ Info:   Loss [1751]: 0.01214
    │         Avg displacement in data: 0.1102
    │         Weight/Scale: 0.753458205232668   Bias/Offset: 0.1998095385635743
    └ @ Main In[14]:15
    ┌ Info:   Loss [1801]: 0.01251
    │         Avg displacement in data: 0.11183
    │         Weight/Scale: 0.7533249040819153   Bias/Offset: 0.2027025314268595
    └ @ Main In[14]:15
    ┌ Info:   Loss [1851]: 0.01214
    │         Avg displacement in data: 0.11019
    │         Weight/Scale: 0.7510613004627918   Bias/Offset: 0.20263583707368665
    └ @ Main In[14]:15
    ┌ Info:   Loss [1901]: 0.01129
    │         Avg displacement in data: 0.10625
    │         Weight/Scale: 0.7461265062862948   Bias/Offset: 0.19691199895980238
    └ @ Main In[14]:15
    ┌ Info:   Loss [1951]: 0.01069
    │         Avg displacement in data: 0.10337
    │         Weight/Scale: 0.7438045369876697   Bias/Offset: 0.19412877849957996
    └ @ Main In[14]:15
    ┌ Info:   Loss [2001]: 0.01081
    │         Avg displacement in data: 0.10398
    │         Weight/Scale: 0.7447911363670117   Bias/Offset: 0.19729867489770186
    └ @ Main In[14]:15
    ┌ Info: Run: 1/2  Epoch: 5/5
    └ @ Main In[26]:3
    ┌ Info:   Loss [2051]: 0.01056
    │         Avg displacement in data: 0.10275
    │         Weight/Scale: 0.7438215153427163   Bias/Offset: 0.1989696740172562
    └ @ Main In[14]:15
    ┌ Info:   Loss [2101]: 0.00913
    │         Avg displacement in data: 0.09554
    │         Weight/Scale: 0.7396131005159379   Bias/Offset: 0.19357734206965932
    └ @ Main In[14]:15
    ┌ Info:   Loss [2151]: 0.00862
    │         Avg displacement in data: 0.09285
    │         Weight/Scale: 0.7360537626846678   Bias/Offset: 0.1881235046568465
    └ @ Main In[14]:15
    ┌ Info:   Loss [2201]: 0.00797
    │         Avg displacement in data: 0.08925
    │         Weight/Scale: 0.7357522170128395   Bias/Offset: 0.1878782265032057
    └ @ Main In[14]:15
    ┌ Info:   Loss [2251]: 0.00739
    │         Avg displacement in data: 0.08596
    │         Weight/Scale: 0.7364570279654614   Bias/Offset: 0.1890258314672458
    └ @ Main In[14]:15
    ┌ Info:   Loss [2301]: 0.00665
    │         Avg displacement in data: 0.08157
    │         Weight/Scale: 0.7359804760205491   Bias/Offset: 0.1882272496760998
    └ @ Main In[14]:15
    ┌ Info:   Loss [2351]: 0.00613
    │         Avg displacement in data: 0.07828
    │         Weight/Scale: 0.7327100312241857   Bias/Offset: 0.18288868822957566
    └ @ Main In[14]:15
    ┌ Info:   Loss [2401]: 0.0055
    │         Avg displacement in data: 0.07416
    │         Weight/Scale: 0.7317309137726691   Bias/Offset: 0.18010566442359519
    └ @ Main In[14]:15
    ┌ Info:   Loss [2451]: 0.0047
    │         Avg displacement in data: 0.06858
    │         Weight/Scale: 0.734141727759607   Bias/Offset: 0.18177505174338934
    └ @ Main In[14]:15
    ┌ Info:   Loss [2501]: 0.00406
    │         Avg displacement in data: 0.0637
    │         Weight/Scale: 0.7360841085762041   Bias/Offset: 0.18313987569650586
    └ @ Main In[14]:15


    1521.568972 seconds (13.85 G allocations: 728.562 GiB, 10.66% gc time, 0.19% compilation time)



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_2.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_3.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_4.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_5.svg)
    


    ┌ Info: Friction model 1 mse: 16.063471558144585
    └ @ Main In[18]:29



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_7.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_8.svg)
    


    ┌ Info: Run: 2/2  Epoch: 1/5
    └ @ Main In[26]:3
    ┌ Info:   Loss [2551]: 0.00574
    │         Avg displacement in data: 0.07574
    │         Weight/Scale: 0.7394674283295187   Bias/Offset: 0.18696360353874708
    └ @ Main In[14]:15
    ┌ Info:   Loss [2601]: 0.00342
    │         Avg displacement in data: 0.05845
    │         Weight/Scale: 0.7424288967630541   Bias/Offset: 0.190979551375045
    └ @ Main In[14]:15
    ┌ Info:   Loss [2651]: 0.00326
    │         Avg displacement in data: 0.05713
    │         Weight/Scale: 0.7437286551818307   Bias/Offset: 0.19317318414384385
    └ @ Main In[14]:15
    ┌ Info:   Loss [2701]: 0.0031
    │         Avg displacement in data: 0.05564
    │         Weight/Scale: 0.744211235516614   Bias/Offset: 0.194257781590303
    └ @ Main In[14]:15
    ┌ Info:   Loss [2751]: 0.00295
    │         Avg displacement in data: 0.05434
    │         Weight/Scale: 0.7448825375255161   Bias/Offset: 0.19564832322821127
    └ @ Main In[14]:15
    ┌ Info:   Loss [2801]: 0.00293
    │         Avg displacement in data: 0.05409
    │         Weight/Scale: 0.7462396660709758   Bias/Offset: 0.198222280616724
    └ @ Main In[14]:15
    ┌ Info:   Loss [2851]: 0.00288
    │         Avg displacement in data: 0.05371
    │         Weight/Scale: 0.7471327344519462   Bias/Offset: 0.20023498491166936
    └ @ Main In[14]:15
    ┌ Info:   Loss [2901]: 0.00289
    │         Avg displacement in data: 0.05379
    │         Weight/Scale: 0.7488109112385878   Bias/Offset: 0.20354612121388982
    └ @ Main In[14]:15
    ┌ Info:   Loss [2951]: 0.00274
    │         Avg displacement in data: 0.05234
    │         Weight/Scale: 0.7497055972126797   Bias/Offset: 0.205819696018191
    └ @ Main In[14]:15
    ┌ Info:   Loss [3001]: 0.00254
    │         Avg displacement in data: 0.05044
    │         Weight/Scale: 0.7479897982290847   Bias/Offset: 0.20419315172920774
    └ @ Main In[14]:15
    ┌ Info: Run: 2/2  Epoch: 2/5
    └ @ Main In[26]:3
    ┌ Info:   Loss [3051]: 0.00225
    │         Avg displacement in data: 0.04746
    │         Weight/Scale: 0.7451072737806487   Bias/Offset: 0.20038343197792163
    └ @ Main In[14]:15
    ┌ Info:   Loss [3101]: 0.00218
    │         Avg displacement in data: 0.04669
    │         Weight/Scale: 0.7426574224064251   Bias/Offset: 0.19734917728526907
    └ @ Main In[14]:15
    ┌ Info:   Loss [3151]: 0.00187
    │         Avg displacement in data: 0.04329
    │         Weight/Scale: 0.742354866981475   Bias/Offset: 0.19717671267933815
    └ @ Main In[14]:15
    ┌ Info:   Loss [3201]: 0.00165
    │         Avg displacement in data: 0.04056
    │         Weight/Scale: 0.7430744542977501   Bias/Offset: 0.198883236946974
    └ @ Main In[14]:15
    ┌ Info:   Loss [3251]: 0.00154
    │         Avg displacement in data: 0.03927
    │         Weight/Scale: 0.7438723827824281   Bias/Offset: 0.2008166136817539
    └ @ Main In[14]:15
    ┌ Info:   Loss [3301]: 0.00153
    │         Avg displacement in data: 0.0391
    │         Weight/Scale: 0.7438480228677498   Bias/Offset: 0.20131434253220434
    └ @ Main In[14]:15
    ┌ Info:   Loss [3351]: 0.00146
    │         Avg displacement in data: 0.03826
    │         Weight/Scale: 0.7449410657035299   Bias/Offset: 0.20298156797745473
    └ @ Main In[14]:15
    ┌ Info:   Loss [3401]: 0.00132
    │         Avg displacement in data: 0.03627
    │         Weight/Scale: 0.7446185517512837   Bias/Offset: 0.20302871829414298
    └ @ Main In[14]:15
    ┌ Info:   Loss [3451]: 0.00131
    │         Avg displacement in data: 0.03622
    │         Weight/Scale: 0.7447316069603234   Bias/Offset: 0.20359839204965716
    └ @ Main In[14]:15
    ┌ Info:   Loss [3501]: 0.00127
    │         Avg displacement in data: 0.03562
    │         Weight/Scale: 0.7440932625523219   Bias/Offset: 0.2033972162354636
    └ @ Main In[14]:15
    ┌ Info: Run: 2/2  Epoch: 3/5
    └ @ Main In[26]:3
    ┌ Info:   Loss [3551]: 0.00128
    │         Avg displacement in data: 0.03578
    │         Weight/Scale: 0.7432924289074272   Bias/Offset: 0.20305658587084313
    └ @ Main In[14]:15
    ┌ Info:   Loss [3601]: 0.00119
    │         Avg displacement in data: 0.03449
    │         Weight/Scale: 0.7429236489307438   Bias/Offset: 0.20326848085161028
    └ @ Main In[14]:15
    ┌ Info:   Loss [3651]: 0.00117
    │         Avg displacement in data: 0.03419
    │         Weight/Scale: 0.7425654634707787   Bias/Offset: 0.20357411966460162
    └ @ Main In[14]:15
    ┌ Info:   Loss [3701]: 0.00109
    │         Avg displacement in data: 0.03294
    │         Weight/Scale: 0.7423634094267662   Bias/Offset: 0.20400466338235348
    └ @ Main In[14]:15
    ┌ Info:   Loss [3751]: 0.00105
    │         Avg displacement in data: 0.03238
    │         Weight/Scale: 0.7421392518583046   Bias/Offset: 0.20448437562924815
    └ @ Main In[14]:15
    ┌ Info:   Loss [3801]: 0.0011
    │         Avg displacement in data: 0.03318
    │         Weight/Scale: 0.7413746723004496   Bias/Offset: 0.20421693127723983
    └ @ Main In[14]:15
    ┌ Info:   Loss [3851]: 0.001
    │         Avg displacement in data: 0.0317
    │         Weight/Scale: 0.7404970565920657   Bias/Offset: 0.20389231161167365
    └ @ Main In[14]:15
    ┌ Info:   Loss [3901]: 0.001
    │         Avg displacement in data: 0.03155
    │         Weight/Scale: 0.7410274386118592   Bias/Offset: 0.20505013042223483
    └ @ Main In[14]:15
    ┌ Info:   Loss [3951]: 0.00092
    │         Avg displacement in data: 0.03026
    │         Weight/Scale: 0.7411430172484441   Bias/Offset: 0.20582198645101407
    └ @ Main In[14]:15
    ┌ Info:   Loss [4001]: 0.00092
    │         Avg displacement in data: 0.03029
    │         Weight/Scale: 0.73994054732983   Bias/Offset: 0.2050838378686791
    └ @ Main In[14]:15
    ┌ Info: Run: 2/2  Epoch: 4/5
    └ @ Main In[26]:3
    ┌ Info:   Loss [4051]: 0.00089
    │         Avg displacement in data: 0.02985
    │         Weight/Scale: 0.7395399400996491   Bias/Offset: 0.2050198345006221
    └ @ Main In[14]:15
    ┌ Info:   Loss [4101]: 0.00095
    │         Avg displacement in data: 0.03088
    │         Weight/Scale: 0.7393288416270515   Bias/Offset: 0.20545447121845742
    └ @ Main In[14]:15
    ┌ Info:   Loss [4151]: 0.00082
    │         Avg displacement in data: 0.02861
    │         Weight/Scale: 0.7402603762008543   Bias/Offset: 0.20706950807045416
    └ @ Main In[14]:15
    ┌ Info:   Loss [4201]: 0.00092
    │         Avg displacement in data: 0.03025
    │         Weight/Scale: 0.7395741717711521   Bias/Offset: 0.20689471136237647
    └ @ Main In[14]:15
    ┌ Info:   Loss [4251]: 0.00082
    │         Avg displacement in data: 0.02866
    │         Weight/Scale: 0.737114361746147   Bias/Offset: 0.20480237341002125
    └ @ Main In[14]:15
    ┌ Info:   Loss [4301]: 0.00078
    │         Avg displacement in data: 0.02786
    │         Weight/Scale: 0.7367630270111356   Bias/Offset: 0.2053714655887574
    └ @ Main In[14]:15
    ┌ Info:   Loss [4351]: 0.00073
    │         Avg displacement in data: 0.02695
    │         Weight/Scale: 0.7376546129263832   Bias/Offset: 0.20716502051184515
    └ @ Main In[14]:15
    ┌ Info:   Loss [4401]: 0.00071
    │         Avg displacement in data: 0.02672
    │         Weight/Scale: 0.7394631033362079   Bias/Offset: 0.21011382986840174
    └ @ Main In[14]:15
    ┌ Info:   Loss [4451]: 0.00068
    │         Avg displacement in data: 0.02617
    │         Weight/Scale: 0.7386981526726747   Bias/Offset: 0.20943313959896703
    └ @ Main In[14]:15
    ┌ Info:   Loss [4501]: 0.00071
    │         Avg displacement in data: 0.02663
    │         Weight/Scale: 0.7398543360977708   Bias/Offset: 0.21018062142602664
    └ @ Main In[14]:15
    ┌ Info: Run: 2/2  Epoch: 5/5
    └ @ Main In[26]:3
    ┌ Info:   Loss [4551]: 0.00067
    │         Avg displacement in data: 0.02593
    │         Weight/Scale: 0.7407214908689865   Bias/Offset: 0.21055511369470342
    └ @ Main In[14]:15
    ┌ Info:   Loss [4601]: 0.00062
    │         Avg displacement in data: 0.02487
    │         Weight/Scale: 0.7410498066117251   Bias/Offset: 0.21061450304937399
    └ @ Main In[14]:15
    ┌ Info:   Loss [4651]: 0.00081
    │         Avg displacement in data: 0.02845
    │         Weight/Scale: 0.7385930602843431   Bias/Offset: 0.20807095572263018
    └ @ Main In[14]:15
    ┌ Info:   Loss [4701]: 0.00059
    │         Avg displacement in data: 0.02419
    │         Weight/Scale: 0.7392744593000797   Bias/Offset: 0.20927363827680145
    └ @ Main In[14]:15
    ┌ Info:   Loss [4751]: 0.0006
    │         Avg displacement in data: 0.02448
    │         Weight/Scale: 0.7398590799013566   Bias/Offset: 0.21092290992559515
    └ @ Main In[14]:15
    ┌ Info:   Loss [4801]: 0.00057
    │         Avg displacement in data: 0.02381
    │         Weight/Scale: 0.7393095559189226   Bias/Offset: 0.21104907855948993
    └ @ Main In[14]:15
    ┌ Info:   Loss [4851]: 0.00056
    │         Avg displacement in data: 0.02363
    │         Weight/Scale: 0.7390001373136241   Bias/Offset: 0.21114028450845262
    └ @ Main In[14]:15
    ┌ Info:   Loss [4901]: 0.00054
    │         Avg displacement in data: 0.02318
    │         Weight/Scale: 0.7389694225582392   Bias/Offset: 0.21148334225196466
    └ @ Main In[14]:15
    ┌ Info:   Loss [4951]: 0.00055
    │         Avg displacement in data: 0.02347
    │         Weight/Scale: 0.7390840093129217   Bias/Offset: 0.21183500472316327
    └ @ Main In[14]:15
    ┌ Info:   Loss [5001]: 0.00054
    │         Avg displacement in data: 0.02321
    │         Weight/Scale: 0.7391624660391095   Bias/Offset: 0.21199520589260332
    └ @ Main In[14]:15


    1163.842974 seconds (10.72 G allocations: 575.871 GiB, 10.71% gc time)



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_11.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_12.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_13.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_14.svg)
    


    ┌ Info: Friction model 1 mse: 16.063471558144585
    └ @ Main In[18]:29
    ┌ Info: Friction model 2 mse: 17.8039866811531
    └ @ Main In[18]:29



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_16.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_17.svg)
    


Finally, the FMU is cleaned-up.


```julia
fmiUnload(simpleFMU)
```

### Summary

Based on the plots, it can be seen that the curves of the *realFMU* and the *neuralFMU* are very close. The *neuralFMU* is able to learn the friction and displacement model.

### Source

[1] Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

