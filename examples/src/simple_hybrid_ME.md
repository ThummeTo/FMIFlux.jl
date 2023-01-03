# Creation and training of ME-NeuralFMUs
Tutorial by Johannes Stoljar, Tobias Thummerer

*Last edit: 03.01.2023*

## License


```julia
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons, Johannes Stoljar
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.
```

## Motivation
The Julia Package *FMIFlux.jl* is motivated by the application of hybrid modeling. This package enables the user to integrate his simulation model between neural networks (NeuralFMU). For this, the simulation model must be exported as FMU (functional mock-up unit), which corresponds to a widely used standard. The big advantage of hybrid modeling with artificial neural networks is, that effects that are difficult to model (because they might be unknown) can be easily learned by the neural networks. For this purpose, the NeuralFMU is trained with measurement data containing the not modeled physical effect. The final product is a simulation model including the originally not modeled effects. Another big advantage of the NeuralFMU is that it works with little data, because the FMU already contains the characteristic functionality of the simulation and only the missing effects are added.

NeuralFMUs do not need to be as easy as in this example. Basically a NeuralFMU can combine different ANN topologies that manipulate any FMU-input (system state, system inputs, time) and any FMU-output (system state derivative, system outputs, other system variables). However, for this example a NeuralFMU topology as shown in the following picture is used.

![NeuralFMU.svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/pics/NeuralFMU.svg?raw=true)

*NeuralFMU (ME) from* [[1]](#Source).

## Introduction to the example
In this example, simplified modeling of a one-dimensional spring pendulum (without friction) is compared to a model of the same system that includes a nonlinear friction model. The FMU with the simplified model will be named *simpleFMU* in the following and the model with the friction will be named *realFMU*. At the beginning, the actual state of both simulations is shown, whereby clear deviations can be seen in the graphs. The *realFMU* serves as a reference graph. The *simpleFMU* is then integrated into a NeuralFMU architecture and a training of the entire network is performed. After the training the final state is compared again to the *realFMU*. It can be clearly seen that by using the NeuralFMU, learning of the friction process has taken place.  


## Target group
The example is primarily intended for users who work in the field of first principle and/or hybrid modeling and are further interested in hybrid model building. The example wants to show how simple it is to combine FMUs with machine learning and to illustrate the advantages of this approach.


## Other formats
Besides, this [Jupyter Notebook](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/simple_hybrid_ME.ipynb) there is also a [Julia file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/simple_hybrid_ME.jl) with the same name, which contains only the code cells and for the documentation there is a [Markdown file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/simple_hybrid_ME.md) corresponding to the notebook.  


## Getting started

### Installation prerequisites
|     | Description                       | Command                   |    
|:----|:----------------------------------|:--------------------------|
| 1.  | Enter Package Manager via         | ]                         |
| 2.  | Install FMI via                   | add FMI                   |
| 3.  | Install FMIFlux via               | add FMIFlux               |
| 4.  | Install FMIZoo via                | add FMIZoo                |
| 5.  | Install DifferentialEquations via | add DifferentialEquations |
| 6.  | Install Plots via                 | add Plots                 |
| 7.  | Install Random via                | add Random                |

## Code section

To run the example, the previously installed packages must be included. 


```julia
# imports
using FMI
using FMIFlux
using FMIZoo
using DifferentialEquations: Tsit5
import Plots

# set seed
import Random
Random.seed!(42);
```

After importing the packages, the path to the *Functional Mock-up Units* (FMUs) is set. The FMU is a model exported meeting the *Functional Mock-up Interface* (FMI) Standard. The FMI is a free standard ([fmi-standard.org](http://fmi-standard.org/)) that defines a container and an interface to exchange dynamic models using a combination of XML files, binaries and C code zipped into a single file. 

The object-orientated structure of the *SpringPendulum1D* (*simpleFMU*) can be seen in the following graphic and corresponds to a simple modeling.

![svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/pics/SpringPendulum1D.svg?raw=true)

In contrast, the model *SpringFrictionPendulum1D* (*realFMU*) is somewhat more accurate, because it includes a friction component. 

![svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/pics/SpringFrictionPendulum1D.svg?raw=true)

Next, the start time and end time of the simulation are set. Finally, a step size is specified to store the results of the simulation at these time steps.


```julia
tStart = 0.0
tStep = 0.01
tStop = 5.0
tSave = collect(tStart:tStep:tStop)
```




    501-element Vector{Float64}:
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
     4.89
     4.9
     4.91
     4.92
     4.93
     4.94
     4.95
     4.96
     4.97
     4.98
     4.99
     5.0



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
    

In the next steps the parameters are defined. The first parameter is the initial position of the mass, which is initilized with $0.5𝑚$. The second parameter is the initial velocity of the mass, which is initialized with $0\frac{m}{s}$. The FMU hase two states: The first state is the position of the mass and the second state is the velocity. In the function fmiSimulate() the *realFMU* is simulated, still specifying the start and end time, the parameters and which variables are recorded. After the simulation is finished the result of the *realFMU* can be plotted. This plot also serves as a reference for the other model (*simpleFMU*).


```julia
initStates = ["s0", "v0"]
x₀ = [0.5, 0.0]
params = Dict(zip(initStates, x₀))
vrs = ["mass.s", "mass.v", "mass.a", "mass.f"]

realSimData = fmiSimulate(realFMU, (tStart, tStop); parameters=params, recordValues=vrs, saveat=tSave)
fmiPlot(realSimData)
```




    
![svg](simple_hybrid_ME_files/simple_hybrid_ME_11_0.svg)
    



The data from the simulation of the *realFMU*, are divided into position and velocity data. These data will be needed later. 


```julia
velReal = fmi2GetSolutionValue(realSimData, "mass.v")
posReal = fmi2GetSolutionValue(realSimData, "mass.s")
```




    501-element Vector{Float64}:
     0.5
     0.5002235448486548
     0.5008715291319449
     0.5019478597521578
     0.5034570452098334
     0.5053993458877354
     0.5077764240578201
     0.5105886522837868
     0.5138351439717114
     0.5175150321322992
     0.521627087567517
     0.5261682148972211
     0.5311370185654775
     ⋮
     1.0657564963384756
     1.066930862706352
     1.0679715872270086
     1.068876303469867
     1.0696434085045978
     1.0702725656148622
     1.0707609890298837
     1.071107075846018
     1.0713093338869186
     1.0713672546639146
     1.0713672546629138
     1.071367254661913



After extracting the data, the FMU is cleaned-up.


```julia
fmiUnload(realFMU)
```

### SimpleFMU

The following lines load, simulate and plot the *simpleFMU* just like the *realFMU*. The differences between both systems can be clearly seen from the plots. In the plot for the *realFMU* it can be seen that the oscillation continues to decrease due to the effect of the friction. If you simulate long enough, the oscillation would come to a standstill in a certain time. The oscillation in the *simpleFMU* behaves differently, since the friction was not taken into account here. The oscillation in this model would continue to infinity with the same oscillation amplitude. From this observation the desire of an improvement of this model arises.     


```julia
simpleFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")
fmiInfo(simpleFMU)

vrs = ["mass.s", "mass.v", "mass.a"]
simpleSimData = fmiSimulate(simpleFMU, (tStart, tStop); recordValues=vrs, saveat=tSave, reset=false)
fmiPlot(simpleSimData)
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
    




    
![svg](simple_hybrid_ME_files/simple_hybrid_ME_17_1.svg)
    



The data from the simulation of the *simpleFMU*, are divided into position and velocity data. These data will be needed later to plot the results. 


```julia
velSimple = fmi2GetSolutionValue(simpleSimData, "mass.v")
posSimple = fmi2GetSolutionValue(simpleSimData, "mass.s")
```




    501-element Vector{Float64}:
     0.5
     0.5003127019074967
     0.5012175433745238
     0.5027172504687035
     0.504812416566759
     0.5075012719497328
     0.5107830165354977
     0.5146534880772458
     0.5191107030735219
     0.5241484264969329
     0.5297629811612266
     0.5359472314461261
     0.5426950964528339
     ⋮
     1.6842615646003007
     1.6884869953422783
     1.6921224800662573
     1.69516502108285
     1.6976144547672483
     1.6994659284032172
     1.7007174453690572
     1.7013675684067706
     1.7014154196220592
     1.7008606804843265
     1.69970552855305
     1.6979508813706



## NeuralFMU

#### Loss function

In order to train our model, a loss function must be implemented. The solver of the NeuralFMU can calculate the gradient of the loss function. The gradient descent is needed to adjust the weights in the neural network so that the sum of the error is reduced and the model becomes more accurate.

The loss function in this implementation consists of the mean squared error (mse) from the real position of the *realFMU* simulation (posReal) and the position data of the network (posNet).
$$ e_{mse} = \frac{1}{n} \sum\limits_{i=0}^n (posReal[i] - posNet[i])^2 $$

As it is indicated with the comments, one could also additionally consider the mse from the real velocity (velReal) and the velocity from the network (velNet). The error in this case would be calculated from the sum of both errors.


```julia
# loss function for training
function lossSum(p)
    global posReal
    solution = neuralFMU(x₀; p=p)

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    FMIFlux.Losses.mse(posReal, posNet) 
end
```




    lossSum (generic function with 1 method)



#### Callback

To output the loss in certain time intervals, a callback is implemented as a function in the following. Here a counter is incremented, every twentieth pass the loss function is called and the average error is printed out.


```julia
# callback function for training
global counter = 0
function callb(p)
    global counter += 1
    if counter % 20 == 1
        avgLoss = lossSum(p[1])
        @info "Loss [$counter]: $(round(avgLoss, digits=5))   Avg displacement in data: $(round(sqrt(avgLoss), digits=5))"
    end
end
```




    callb (generic function with 1 method)



#### Structure of the NeuralFMU

In the following, the topology of the NeuralFMU is constructed. It consists of an input layer, which then leads into the *simpleFMU* model. The ME-FMU computes the state derivatives for a given system state. Following the *simpleFMU* is a dense layer that has exactly as many inputs as the model has states (and therefore state derivatives). The output of this layer consists of 16 output nodes and a *tanh* activation function. The next layer has 16 input and output nodes with the same activation function. The last layer is again a dense layer with 16 input nodes and the number of states as outputs. Here, it is important that no *tanh*-activation function follows, because otherwise the pendulums state values would be limited to the interval $[-1;1]$.


```julia
# NeuralFMU setup
numStates = fmiGetNumberOfStates(simpleFMU)

net = Chain(inputs -> fmiEvaluateME(simpleFMU, inputs),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))
```




    Chain(
      var"#1#2"(),
      Dense(2 => 16, tanh),                 [90m# 48 parameters[39m
      Dense(16 => 16, tanh),                [90m# 272 parameters[39m
      Dense(16 => 2),                       [90m# 34 parameters[39m
    ) [90m                  # Total: 6 arrays, [39m354 parameters, 3.141 KiB.



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




    
![svg](simple_hybrid_ME_files/simple_hybrid_ME_29_0.svg)
    



#### Training of the NeuralFMU

For the training of the NeuralFMU the parameters are extracted. The known Adam optimizer for minimizing the gradient descent is used as further passing parameters. In addition, the previously defined loss and callback function, as well as the number of epochs are passed.


```julia
# train
paramsNet = FMIFlux.params(neuralFMU)

optim = Adam()
FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), 300), optim; cb=()->callb(paramsNet)) 
```

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1]: 22.23882   Avg displacement in data: 4.71581
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [21]: 2.24523   Avg displacement in data: 1.49841
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [41]: 0.09837   Avg displacement in data: 0.31364
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [61]: 0.05964   Avg displacement in data: 0.24422
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [81]: 0.05325   Avg displacement in data: 0.23077
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [101]: 0.05016   Avg displacement in data: 0.22397
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [121]: 0.04782   Avg displacement in data: 0.21867
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [141]: 0.04585   Avg displacement in data: 0.21413
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [161]: 0.04416   Avg displacement in data: 0.21013
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [181]: 0.04268   Avg displacement in data: 0.20658
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [201]: 0.04137   Avg displacement in data: 0.2034
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [221]: 0.04021   Avg displacement in data: 0.20052
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [241]: 0.03917   Avg displacement in data: 0.19792
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [261]: 0.03824   Avg displacement in data: 0.19556
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [281]: 0.03739   Avg displacement in data: 0.19338
    

#### Comparison of the plots

Here three plots are compared with each other and only the position of the mass is considered. The first plot represents the *simpleFMU*, the second represents the *realFMU* (reference) and the third plot represents the result after training the NeuralFMU. 


```julia
# plot results mass.s
solutionAfter = neuralFMU(x₀)

fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
                 xtickfontsize=12, ytickfontsize=12,
                 xguidefontsize=12, yguidefontsize=12,
                 legendfontsize=8, legend=:topright)

posNeuralFMU = fmi2GetSolutionState(solutionAfter, 1; isIndex=true)

Plots.plot!(fig, tSave, posSimple, label="SimpleFMU", linewidth=2)
Plots.plot!(fig, tSave, posReal, label="RealFMU", linewidth=2)
Plots.plot!(fig, tSave, posNeuralFMU, label="NeuralFMU (300 epochs)", linewidth=2)
fig 
```




    
![svg](simple_hybrid_ME_files/simple_hybrid_ME_33_0.svg)
    



#### Continue training and plotting

As can be seen from the previous figure, the plot of the NeuralFMU has not yet fully converged against the *realFMU*, so the training of the NeuralFMU is continued. After further training, the plot of *NeuralFMU* is added to the figure again. The effect of the longer training can be recognized well, since the plot of the NeuralFMU had further converged. 


```julia
FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), 1200), optim; cb=()->callb(paramsNet)) 
# plot results mass.s
solutionAfter = neuralFMU(x₀)
posNeuralFMU = fmi2GetSolutionState(solutionAfter, 1; isIndex=true)
Plots.plot!(fig, tSave, posNeuralFMU, label="NeuralFMU (1500 epochs)", linewidth=2)
fig 
```

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [301]: 0.0367   Avg displacement in data: 0.19158
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [321]: 0.03607   Avg displacement in data: 0.18992
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [341]: 0.03551   Avg displacement in data: 0.18845
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [361]: 0.03502   Avg displacement in data: 0.18715
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [381]: 0.03459   Avg displacement in data: 0.18598
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [401]: 0.0342   Avg displacement in data: 0.18494
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [421]: 0.03385   Avg displacement in data: 0.18397
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [441]: 0.03351   Avg displacement in data: 0.18306
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [461]: 0.03319   Avg displacement in data: 0.18217
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [481]: 0.03286   Avg displacement in data: 0.18128
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [501]: 0.03253   Avg displacement in data: 0.18035
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [521]: 0.03218   Avg displacement in data: 0.17938
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [541]: 0.03181   Avg displacement in data: 0.17835
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [561]: 0.03141   Avg displacement in data: 0.17723
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [581]: 0.03098   Avg displacement in data: 0.17602
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [601]: 0.03052   Avg displacement in data: 0.1747
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [621]: 0.03002   Avg displacement in data: 0.17326
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [641]: 0.02948   Avg displacement in data: 0.17169
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [661]: 0.02889   Avg displacement in data: 0.16998
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [681]: 0.02827   Avg displacement in data: 0.16814
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [701]: 0.02758   Avg displacement in data: 0.16608
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [721]: 0.02685   Avg displacement in data: 0.16385
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [741]: 0.02607   Avg displacement in data: 0.16146
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [761]: 0.02524   Avg displacement in data: 0.15887
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [781]: 0.02435   Avg displacement in data: 0.15605
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [801]: 0.0234   Avg displacement in data: 0.15296
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [821]: 0.02236   Avg displacement in data: 0.14954
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [841]: 0.02123   Avg displacement in data: 0.1457
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [861]: 0.01999   Avg displacement in data: 0.14138
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [881]: 0.01864   Avg displacement in data: 0.13654
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [901]: 0.01724   Avg displacement in data: 0.1313
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [921]: 0.01583   Avg displacement in data: 0.12581
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [941]: 0.01446   Avg displacement in data: 0.12024
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [961]: 0.01316   Avg displacement in data: 0.11473
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [981]: 0.01198   Avg displacement in data: 0.10944
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1001]: 0.01093   Avg displacement in data: 0.10452
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1021]: 0.00995   Avg displacement in data: 0.09976
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1041]: 0.00899   Avg displacement in data: 0.09481
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1061]: 0.00803   Avg displacement in data: 0.08963
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1081]: 0.00716   Avg displacement in data: 0.08462
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1101]: 0.00644   Avg displacement in data: 0.08027
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1121]: 0.00593   Avg displacement in data: 0.07698
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1141]: 0.0056   Avg displacement in data: 0.07482
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1161]: 0.0053   Avg displacement in data: 0.07277
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1181]: 0.00516   Avg displacement in data: 0.07181
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1201]: 0.00505   Avg displacement in data: 0.07104
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1221]: 0.00495   Avg displacement in data: 0.07035
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1241]: 0.00485   Avg displacement in data: 0.06961
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1261]: 0.00475   Avg displacement in data: 0.06894
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1281]: 0.00473   Avg displacement in data: 0.06875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1301]: 0.00459   Avg displacement in data: 0.06775
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1321]: 0.00456   Avg displacement in data: 0.06753
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1341]: 0.0045   Avg displacement in data: 0.06709
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1361]: 0.00438   Avg displacement in data: 0.06616
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1381]: 0.00428   Avg displacement in data: 0.06544
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1401]: 0.00422   Avg displacement in data: 0.06495
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1421]: 0.00415   Avg displacement in data: 0.06446
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1441]: 0.0041   Avg displacement in data: 0.06405
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1461]: 0.00402   Avg displacement in data: 0.06343
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mLoss [1481]: 0.00401   Avg displacement in data: 0.06334
    




    
![svg](simple_hybrid_ME_files/simple_hybrid_ME_35_1.svg)
    



Finally, the FMU is cleaned-up.


```julia
fmiUnload(simpleFMU)
```

### Summary

Based on the plots, it can be seen that the NeuralFMU is able to adapt the friction model of the *realFMU*. After 300 runs, the curves do not overlap very well, but this can be achieved by longer training (1000 runs) or a better initialization.

### Source

[1] Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

