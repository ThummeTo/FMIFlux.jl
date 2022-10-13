# Creation and training of ME-NeuralFMUs
Tutorial by Johannes Stoljar, Tobias Thummerer

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
Besides, this [Jupyter Notebook](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/simple_hybrid_ME.ipynb) there is also a [Julia file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/simple_hybrid_ME.jl) with the same name, which contains only the code cells and for the documentation there is a [Markdown file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/simple_hybrid_ME.md) corresponding to the notebook.  


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

    ┌ Info: fmi2Unzip(...): Successfully unzipped 153 files at `/tmp/fmijl_WU0Dki/SpringFrictionPendulum1D`.
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/1Yngw/src/FMI2_ext.jl:90
    ┌ Info: fmi2Load(...): FMU resources location is `file:////tmp/fmijl_WU0Dki/SpringFrictionPendulum1D/resources`
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/1Yngw/src/FMI2_ext.jl:221
    ┌ Info: fmi2Load(...): FMU supports both CS and ME, using CS as default if nothing specified.
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/1Yngw/src/FMI2_ext.jl:224


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

realSimData = fmiSimulate(realFMU, tStart, tStop; parameters=params, recordValues=vrs, saveat=tSave)
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
     1.0657564963230959
     1.0669308626658962
     1.0679715871563396
     1.0688763033630924
     1.069643408396874
     1.070272565475436
     1.0707609888490195
     1.0711070756109329
     1.0713093335814168
     1.0713672542789878
     1.071367254277987
     1.0713672542769863



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
simpleSimData = fmiSimulate(simpleFMU, tStart, tStop; recordValues=vrs, saveat=tSave, reset=false)
fmiPlot(simpleSimData)
```

    ┌ Info: fmi2Unzip(...): Successfully unzipped 153 files at `/tmp/fmijl_WsxZ1h/SpringPendulum1D`.
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/1Yngw/src/FMI2_ext.jl:90
    ┌ Info: fmi2Load(...): FMU resources location is `file:////tmp/fmijl_WsxZ1h/SpringPendulum1D/resources`
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/1Yngw/src/FMI2_ext.jl:221


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


    ┌ Info: fmi2Load(...): FMU supports both CS and ME, using CS as default if nothing specified.
    └ @ FMIImport /home/runner/.julia/packages/FMIImport/1Yngw/src/FMI2_ext.jl:224





    
![svg](simple_hybrid_ME_files/simple_hybrid_ME_17_3.svg)
    



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
$$ mse = \frac{1}{n} \sum\limits_{i=0}^n (posReal[i] - posNet[i])^2 $$

As it is indicated with the comments, one could also additionally consider the mse from the real velocity (velReal) and the velocity from the network (velNet). The error in this case would be calculated from the sum of both errors.


```julia
# loss function for training
function lossSum(p)
    global posReal
    solution = neuralFMU(x₀, tStart; p=p)

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    # velNet = fmi2GetSolutionState(solution, 2; isIndex=true)

    Flux.Losses.mse(posReal, posNet) #+ Flux.Losses.mse(velReal, velNet)
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
    ) [90m                  # Total: 6 arrays, [39m354 parameters, 1.758 KiB.



#### Definition of the NeuralFMU

The instantiation of the ME-NeuralFMU is done as a one-liner. The FMU (*simpleFMU*), the structure of the network `net`, start `tStart` and end time `tStop`, the numerical solver `Tsit5()` and the time steps `tSave` for saving are specified.


```julia
neuralFMU = ME_NeuralFMU(simpleFMU, net, (tStart, tStop), Tsit5(); saveat=tSave);
```

    ┌ Info: ME_NeuralFMU(...): Succesfully converted layer of type `Dense{typeof(tanh), Matrix{Float32}, Vector{Float32}}` to `Dense{typeof(tanh), Matrix{Float64}, Vector{Float64}}`.
    └ @ FMIFlux /home/runner/.julia/packages/FMIFlux/WeryT/src/FMI_neural.jl:1089
    ┌ Info: ME_NeuralFMU(...): Succesfully converted layer of type `Dense{typeof(tanh), Matrix{Float32}, Vector{Float32}}` to `Dense{typeof(tanh), Matrix{Float64}, Vector{Float64}}`.
    └ @ FMIFlux /home/runner/.julia/packages/FMIFlux/WeryT/src/FMI_neural.jl:1089
    ┌ Info: ME_NeuralFMU(...): Succesfully converted layer of type `Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}` to `Dense{typeof(identity), Matrix{Float64}, Vector{Float64}}`.
    └ @ FMIFlux /home/runner/.julia/packages/FMIFlux/WeryT/src/FMI_neural.jl:1089
    ┌ Info: ME_NeuralFMU(...): Succesfully converted model to Float64.
    └ @ FMIFlux /home/runner/.julia/packages/FMIFlux/WeryT/src/FMI_neural.jl:1103


#### Plot before training

Here the state trajectory of the *simpleFMU* is recorded. Doesn't really look like a pendulum yet, but the system is random initialized by default. In the plots later on, the effect of learning can be seen.


```julia
solutionBefore = neuralFMU(x₀, tStart)
fmiPlot(solutionBefore)
```

    [34mSimulating ME-NeuralFMU ... 100%|████████████████████████| Time: 0:00:20[39m





    
![svg](simple_hybrid_ME_files/simple_hybrid_ME_29_1.svg)
    



#### Training of the NeuralFMU

For the training of the NeuralFMU the parameters are extracted. The known ADAM optimizer for minimizing the gradient descent is used as further passing parameters. In addition, the previously defined loss and callback function, as well as the number of epochs are passed.


```julia
# train
paramsNet = Flux.params(neuralFMU)

optim = ADAM()
FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), 300), optim; cb=()->callb(paramsNet)) 
```

    [34mSimulating ME-NeuralFMU ... 100%|████████████████████████| Time: 0:00:30[39m
    ┌ Info: Loss [1]: 0.76769   Avg displacement in data: 0.87618
    └ @ Main In[11]:7
    ┌ Info: Loss [21]: 0.06822   Avg displacement in data: 0.2612
    └ @ Main In[11]:7
    ┌ Info: Loss [41]: 0.05321   Avg displacement in data: 0.23066
    └ @ Main In[11]:7
    ┌ Info: Loss [61]: 0.04778   Avg displacement in data: 0.2186
    └ @ Main In[11]:7
    ┌ Info: Loss [81]: 0.04434   Avg displacement in data: 0.21057
    └ @ Main In[11]:7
    ┌ Info: Loss [101]: 0.04209   Avg displacement in data: 0.20516
    └ @ Main In[11]:7
    ┌ Info: Loss [121]: 0.04092   Avg displacement in data: 0.20228
    └ @ Main In[11]:7
    ┌ Info: Loss [141]: 0.04032   Avg displacement in data: 0.20079
    └ @ Main In[11]:7
    ┌ Info: Loss [161]: 0.03989   Avg displacement in data: 0.19974
    └ @ Main In[11]:7
    ┌ Info: Loss [181]: 0.03957   Avg displacement in data: 0.19893
    └ @ Main In[11]:7
    ┌ Info: Loss [201]: 0.03933   Avg displacement in data: 0.19832
    └ @ Main In[11]:7
    ┌ Info: Loss [221]: 0.03914   Avg displacement in data: 0.19784
    └ @ Main In[11]:7
    ┌ Info: Loss [241]: 0.03899   Avg displacement in data: 0.19746
    └ @ Main In[11]:7
    ┌ Info: Loss [261]: 0.03887   Avg displacement in data: 0.19715
    └ @ Main In[11]:7
    ┌ Info: Loss [281]: 0.03876   Avg displacement in data: 0.19689
    └ @ Main In[11]:7


#### Comparison of the plots

Here three plots are compared with each other and only the position of the mass is considered. The first plot represents the *simpleFMU*, the second represents the *realFMU* (reference) and the third plot represents the result after training the NeuralFMU. 


```julia
# plot results mass.s
solutionAfter = neuralFMU(x₀, tStart)

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
FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), 700), optim; cb=()->callb(paramsNet)) 
# plot results mass.s
solutionAfter = neuralFMU(x₀, tStart)
posNeuralFMU = fmi2GetSolutionState(solutionAfter, 1; isIndex=true)
Plots.plot!(fig, tSave, posNeuralFMU, label="NeuralFMU (1000 epochs)", linewidth=2)
fig 
```

    ┌ Info: Loss [301]: 0.03868   Avg displacement in data: 0.19666
    └ @ Main In[11]:7
    ┌ Info: Loss [321]: 0.0386   Avg displacement in data: 0.19646
    └ @ Main In[11]:7
    ┌ Info: Loss [341]: 0.03853   Avg displacement in data: 0.19628
    └ @ Main In[11]:7
    ┌ Info: Loss [361]: 0.03846   Avg displacement in data: 0.19611
    └ @ Main In[11]:7
    ┌ Info: Loss [381]: 0.03839   Avg displacement in data: 0.19594
    └ @ Main In[11]:7
    ┌ Info: Loss [401]: 0.03833   Avg displacement in data: 0.19578
    └ @ Main In[11]:7
    ┌ Info: Loss [421]: 0.03826   Avg displacement in data: 0.19561
    └ @ Main In[11]:7
    ┌ Info: Loss [441]: 0.0382   Avg displacement in data: 0.19544
    └ @ Main In[11]:7
    ┌ Info: Loss [461]: 0.03813   Avg displacement in data: 0.19527
    └ @ Main In[11]:7
    ┌ Info: Loss [481]: 0.03806   Avg displacement in data: 0.19509
    └ @ Main In[11]:7
    ┌ Info: Loss [501]: 0.03798   Avg displacement in data: 0.19489
    └ @ Main In[11]:7
    ┌ Info: Loss [521]: 0.0379   Avg displacement in data: 0.19468
    └ @ Main In[11]:7
    ┌ Info: Loss [541]: 0.03781   Avg displacement in data: 0.19446
    └ @ Main In[11]:7
    ┌ Info: Loss [561]: 0.03772   Avg displacement in data: 0.19421
    └ @ Main In[11]:7
    ┌ Info: Loss [581]: 0.03761   Avg displacement in data: 0.19394
    └ @ Main In[11]:7
    ┌ Info: Loss [601]: 0.0375   Avg displacement in data: 0.19365
    └ @ Main In[11]:7
    ┌ Info: Loss [621]: 0.03737   Avg displacement in data: 0.19332
    └ @ Main In[11]:7
    ┌ Info: Loss [641]: 0.03723   Avg displacement in data: 0.19295
    └ @ Main In[11]:7
    ┌ Info: Loss [661]: 0.03707   Avg displacement in data: 0.19252
    └ @ Main In[11]:7
    ┌ Info: Loss [681]: 0.03689   Avg displacement in data: 0.19207
    └ @ Main In[11]:7
    ┌ Info: Loss [701]: 0.03667   Avg displacement in data: 0.1915
    └ @ Main In[11]:7
    ┌ Info: Loss [721]: 0.03644   Avg displacement in data: 0.19089
    └ @ Main In[11]:7
    ┌ Info: Loss [741]: 0.03616   Avg displacement in data: 0.19015
    └ @ Main In[11]:7
    ┌ Info: Loss [761]: 0.03582   Avg displacement in data: 0.18927
    └ @ Main In[11]:7
    ┌ Info: Loss [781]: 0.03542   Avg displacement in data: 0.18819
    └ @ Main In[11]:7
    ┌ Info: Loss [801]: 0.03492   Avg displacement in data: 0.18686
    └ @ Main In[11]:7
    ┌ Info: Loss [821]: 0.03429   Avg displacement in data: 0.18518
    └ @ Main In[11]:7
    ┌ Info: Loss [841]: 0.03348   Avg displacement in data: 0.18296
    └ @ Main In[11]:7
    ┌ Info: Loss [861]: 0.03238   Avg displacement in data: 0.17995
    └ @ Main In[11]:7
    ┌ Info: Loss [881]: 0.03084   Avg displacement in data: 0.17561
    └ @ Main In[11]:7
    ┌ Info: Loss [901]: 0.02854   Avg displacement in data: 0.16893
    └ @ Main In[11]:7
    ┌ Info: Loss [921]: 0.02494   Avg displacement in data: 0.15792
    └ @ Main In[11]:7
    ┌ Info: Loss [941]: 0.01953   Avg displacement in data: 0.13974
    └ @ Main In[11]:7
    ┌ Info: Loss [961]: 0.01308   Avg displacement in data: 0.11436
    └ @ Main In[11]:7
    ┌ Info: Loss [981]: 0.00879   Avg displacement in data: 0.09375
    └ @ Main In[11]:7





    
![svg](simple_hybrid_ME_files/simple_hybrid_ME_35_1.svg)
    



Finally, the FMU is cleaned-up.


```julia
fmiUnload(simpleFMU)
```

### Summary

Based on the plots, it can be seen that the NeuralFMU is able to adapt the friction model of the *realFMU*. After 300 runs, the curves do not overlap very well, but this can be achieved by longer training (1000 runs) or a better initialization.

### Source

[1] Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

