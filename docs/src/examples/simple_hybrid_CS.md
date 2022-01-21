# Creation and training of CS-NeuralFMUs
Tutorial by Johannes Stoljar, Tobias Thummerer

## License
Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons, Johannes Stoljar

Licensed under the MIT license. See [LICENSE](https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

## Motivation
This Julia Package is motivated by the application of hybrid modeling. This package enables the user to integrate his simulation model between neural networks (NeuralFMU). For this, the simulation model must be exported as FMU (functional mock-up unit), which corresponds to a widely used standard. The big advantage of hybrid modeling with artificial neural networks is, that effects that are difficult to model (because they might be unknown) can be easily learned by the neural networks. For this purpose, the NeuralFMU is trained with measurement data containing the unmodeled physical effect. The final product is a simulation model including the orignially unmodeled effects. Another big advantage of the NeuralFMU is that it works with little data, because the FMU already contains the characterisitic functionality of the simulation and only the missing effects are added.

NeuralFMUs need not to be as easy as in this example. Basically a NeuralFMU can combine different ANN topologies that manipulate any FMU-input (system state, system inputs, time) and any FMU-output (system state derivative, system outputs, other system variables). However, for this example a NeuralFMU topology as shown in the following picture is used.

![CS-NeuralFMU.svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/pics/CSNeuralFMU.svg?raw=true)

*NeuralFMU (CS) from* [[1]](#Source).

## Introduction to the example
In this example, the model of a one-dimensional spring pendulum (with an external acting force) is used to learn the initial states. For this purpose, on the one hand the initial position of the mass of the pendulum is shifted and on the other hand the default position of the mass from the model is used. The model with the shifted initial position serves as reference and is called *referenceFmu* in the following. The model with the default position is further referenced with *defaultFmu*. At the beginning, the actual state of both simulations is shown, whereby clear deviations can be seen in the graphs. Afterwards, the *defaultFmu* is integrated into a co-simulation NeuralFMU (CS-NeuralFMU) architecture. By training the NeuralFMU, an attempt is made to learn the initial displacement of the *referenceFMU*. It can be clearly seen that the NeuralFMU learns this shift well in just a few training steps. 


## Target group
The example is primarily intended for users who work in the field of first principle and/or hybrid modeling and are further interested in hybrid model building. The example wants to show how simple it is to combine FMUs with machine learning and to illustrate the advantages of this approach.


## Other formats
Besides this [Jupyter Notebook](https://github.com/thummeto/FMIFlux.jl/blob/main/example/simple_hybrid_CS.ipynb) there is also a [Julia file](https://github.com/thummeto/FMIFlux.jl/blob/main/example/simple_hybrid_CS.jl) with the same name, which contains only the code cells and for the documentation there is a [Markdown file](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/simple_hybrid_CS.md) corresponding to the notebook.  


## Getting started

### Installation prerequisites
|    | Description                       | Command     |  Alternative  |   
|:--- |:---                               |:---        |:---|
|1.  | Enter Package Manager via         |     ]       |     |
|2.  | Install FMI via                   |   add FMI   | add " https://github.com/ThummeTo/FMI.jl "   |
|3.  | Install FMIFlux via               | add FMIFlux | add " https://github.com/ThummeTo/FMIFlux.jl " |
|4.  | Install Flux via                  |  add Flux   |     |
|5.  | Install DifferentialEquations via | add DifferentialEquations |  |
|6.  | Install Plots via                 | add Plots   |     |

## Code section

To run the example, the previously installed packages must be included. 


```julia
# imports
using FMI
using FMIFlux
using Flux
using DifferentialEquations: Tsit5
import Plots
```

After importing the packages, the path to the *Functional Mock-up Units* (FMUs) is set. The FMU is a model exported meeting the *Functional Mock-up Interface* (FMI) Standard. The FMI is a free standard ([fmi-standard.org](http://fmi-standard.org/)) that defines a container and an interface to exchange dynamic models using a combination of XML files, binaries and C code zipped into a single file. 

The objec-orientated structure of the *SpringPendulumExtForce1D* can be seen in the following graphic. This model is a simple spring pendulum without friction, but with an external force. 

![svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/pics/SpringPendulumExtForce1D.svg?raw=true)

Here the path for the [*SpringPendulumExtForce1D*](https://github.com/thummeto/FMIFlux.jl/blob/main/model/SpringPendulumExtForce1D.fmu) is set: 



```julia
fmuPath = joinpath(dirname(@__FILE__), "../model/SpringPendulumExtForce1D.fmu")
println("FMU path: ", fmuPath)
```

    FMU path: ../model/SpringPendulumExtForce1D.fmu


Next, the start time and end time of the simulation are set. Finally, a step size is specified to store the results of the simulation at these time steps.


```julia
tStart = 0.0
tStep = 0.01
tStop = 5.0
tSave = tStart:tStep:tStop
```




    0.0:0.01:5.0



### ReferenceFmu

In the next lines of code the FMU of the *referenceFmu* model is loaded and instantiated.  


```julia
referenceFmu = fmiLoad(fmuPath)
fmiInstantiate!(referenceFmu; loggingOn=false)
fmiInfo(referenceFmu)
```

    ┌ Info: fmi2Unzip(...): Successfully unzipped 147 files at `C:\Users\JOHANN~1\AppData\Local\Temp\fmijl_WJMFpf\SpringPendulumExtForce1D`.
    └ @ FMI C:\Users\Johannes Stoljar\.julia\packages\FMI\l4qPg\src\FMI2.jl:273
    ┌ Info: fmi2Load(...): FMU supports both CS and ME, using CS as default if nothing specified.
    └ @ FMI C:\Users\Johannes Stoljar\.julia\packages\FMI\l4qPg\src\FMI2.jl:376
    ┌ Info: fmi2Load(...): FMU resources location is `file:///C:/Users/JOHANN~1/AppData/Local/Temp/fmijl_WJMFpf/SpringPendulumExtForce1D/resources`
    └ @ FMI C:\Users\Johannes Stoljar\.julia\packages\FMI\l4qPg\src\FMI2.jl:384


    #################### Begin information for FMU ####################
    	Model name:			SpringPendulumExtForce1D
    	FMI-Version:			2.0
    	GUID:				{b376bbba-5027-4429-a701-20b703fda94e}
    	Generation tool:		Dymola Version 2020x (64-bit), 2019-10-10
    	Generation time:		2021-06-18T11:01:53Z
    	Var. naming conv.:		structured
    	Event indicators:		0
    	Inputs:				1
    		352321536 ["extForce"]
    	Outputs:			2
    		335544320 ["der(accSensor.v)", "a", "accSensor.a"]
    		335544321 ["accSensor.v", "der(accSensor.flange.s)", "v", "der(speedSensor.flange.s)", "speedSensor.v"]
    	States:				2
    		33554432 ["mass.s"]
    		33554433 ["mass.v"]
    	Supports Co-Simulation:		true
    		Model identifier:	SpringPendulumExtForce1D
    		Get/Set State:		true
    		Serialize State:	true
    		Dir. Derivatives:	true
    		Var. com. steps:	true
    		Input interpol.:	true
    		Max order out. der.:	1
    	Supports Model-Exchange:	true
    		Model identifier:	SpringPendulumExtForce1D
    		Get/Set State:		true
    		Serialize State:	true
    		Dir. Derivatives:	true
    ##################### End information for FMU #####################


Both the start and end time are set via the *fmiSetupExperiment()* function. In addition, the initial position of the mass is set to a value of $1.3m$  The experiment is initialized to get the information of the continuous states. You can get all continuous states of a FMU by the function *fmiGetContinuousStates()* and this is also done for the *referenceFmu*. It has two states: The first state is the previously initialized position of the mass, the second state is the velocity, which is initialized with $0\frac{m}{s}$.   


```julia
fmiSetupExperiment(referenceFmu, tStart, tStop)
fmiSetReal(referenceFmu, "mass_s0", 1.3)   # increase amplitude, invert phase
fmiEnterInitializationMode(referenceFmu)
fmiExitInitializationMode(referenceFmu)

x₀ = fmiGetContinuousStates(referenceFmu)
```




    2-element Vector{Float64}:
     1.3
     0.0



In the following code block the *referenceFmu* is simulated, still specifying which variables are included. After the simulation is finished the result of the *referenceFmu* can be plotted. This plot also serves as a reference for the later CS-NeuralFMU model.


```julia
vrs = ["mass.s", "mass.v", "mass.a"]
_, referenceSimData = fmiSimulate(referenceFmu, tStart, tStop; recordValues=vrs, setup=false, reset=false, saveat=tSave)
fmiPlot(referenceFmu, vrs, referenceSimData)
```




    
![svg](simple_hybrid_CS_files/simple_hybrid_CS_12_0.svg)
    



The data from the simualtion of the *referenceFmu*, are divided into position, velocity and acceleration data. The data for the acceleration will be needed later. 


```julia
posReference = collect(data[1] for data in referenceSimData.saveval)
velReference = collect(data[2] for data in referenceSimData.saveval)
accReference = collect(data[3] for data in referenceSimData.saveval)
```




    501-element Vector{Float64}:
     -1.9999999999999996
     -1.9989808107156004
     -1.995976332371232
     -1.9909821938997307
     -1.9839989801021418
     -1.9750314004124547
     -1.9640884504035183
     -1.951180580066516
     -1.9363227134824257
     -1.9195319560818125
     -1.9008203166719828
     -1.8802131771552166
     -1.8577245801802755
      ⋮
      1.9439538472626405
      1.9581269688364755
      1.970346615172437
      1.9805952930006132
      1.9888623187994514
      1.9951388459819808
      1.9994178648958127
      2.0016968375647415
      2.0019759530917005
      2.0002523498984894
      1.9965275218318568
      1.9908049090723823



### DefaultFmu

The following is a reset for the *referenceFmu* and a renaming to *defaultFmu*. After the reset, the previous initial position of the mass is not set, so the default position of the *defaultFmu* is used. The first state indicates the position of the mass, which is initilized with $0.5𝑚$.


```julia
fmiReset(referenceFmu)
defaultFmu = referenceFmu

fmiSetupExperiment(defaultFmu, tStart, tStop)
fmiEnterInitializationMode(defaultFmu)
fmiExitInitializationMode(defaultFmu)

x₀ = fmiGetContinuousStates(defaultFmu)
```




    2-element Vector{Float64}:
     0.5
     0.0



The following simulate and plot the *defaultFmu* just like the *referenceFmu*. The differences between both systems can be clearly seen from the plots. In the plots for the *defaultFmu* you can see that other oscillations occur due to the different starting positions. On the one hand the oscillation of the *defaultFmu* starts in the opposite direction of the *referenceFmu* and on the other hand the graphs for the velocity and acceleration differ clearly in the amplitude. In the following we try to learn the initial shift of the position so that the graphs for the acceleration of both graphs match.


```julia
_, defaultSimData = fmiSimulate(defaultFmu, tStart, tStop; recordValues=vrs, setup=false, reset=false, saveat=tSave)
fmiPlot(defaultFmu, vrs, defaultSimData)
```




    
![svg](simple_hybrid_CS_files/simple_hybrid_CS_18_0.svg)
    



The data from the simualtion of the *defaultFmu*, are divided into position, velocity and acceleration data. The data for the acceleration will be needed later.


```julia
posDefault = collect(data[1] for data in defaultSimData.saveval)
velDefault = collect(data[2] for data in defaultSimData.saveval)
accDefault = collect(data[3] for data in defaultSimData.saveval)
```




    501-element Vector{Float64}:
      6.0
      5.996982513180007
      5.987986261034271
      5.9730046030442665
      5.95205107717745
      5.925151171646224
      5.892330901036602
      5.853619029884401
      5.809060133574773
      5.758686483925127
      5.702562314755022
      5.640726974472335
      5.5732600661315335
      ⋮
     -5.817454106640481
     -5.860262621029506
     -5.897211631373532
     -5.928264987575014
     -5.953392775960465
     -5.972564609104964
     -5.985762763815119
     -5.99297561097946
     -5.994196580640214
     -5.989425410415006
     -5.9786675103892755
     -5.961926527257058



## CS-NeuralFMU

In this section, the *defaultFmu* is inserted into a CS-NeuralFMU architecture. It has the goal to learn the initial state of the *referenceFmu*.


For the external force, a simple function is implemented that always returns a force of $0N$ at each time point. Also, all other functions and implementations would be possible here. Only for simplification reasons the function was chosen so simply.


```julia
function extForce(t)
    return [0.0]
end 
```




    extForce (generic function with 1 method)



#### Loss function

In order to train our model, a loss function must be implemented. The solver of the NeuralFMU can calculate the gradient of the loss function. The gradient descent is needed to adjust the weights in the neural network so that the sum of the error is reduced and the model becomes more accurate.

The loss function in this implmentation consists of the mean squared error (mse) from the acceleration data of the *referenceFmu* simulation (`accReference`) and the acceleration data of the network (`accNet`).
$$ mse = \frac{1}{n} \sum\limits_{i=0}^n (accReference[i] - accNet[i])^2 $$


```julia
# loss function for training
function lossSum()
    solution = csNeuralFmu(extForce, tStep)

    accNet = collect(data[1] for data in solution)
    
    Flux.Losses.mse(accReference, accNet)
end
```




    lossSum (generic function with 1 method)



#### Callback

To output the loss in certain time intervals, a callback is implemented as a function in the following. Here a counter is incremented, every twentieth pass the loss function is called and the average error is printed out.


```julia
# callback function for training
global counter = 0
function callb()
    global counter += 1

    if counter % 20 == 1
        avgLoss = lossSum()
        @info "Loss [$counter]: $(round(avgLoss, digits=5))"
    end
end
```




    callb (generic function with 1 method)



#### Structure of the CS-NeuralFMU

In the following, the topology of the CS-NeuralFMU is constructed. It consists of an input layer, which then leads into the *defaultFmu* model. The CS-FMU computes the outputs for the given system state and time step. After the *defaultFmu* follows a dense layer, which has exactly as many inputs as the model has outputs. The output of this layer consists of 16 output nodes and a *tanh* activation function. The next layer has 16 input and output nodes with the same activation function. The last layer is again a dense layer with 16 input nodes and the number of model outputs as output nodes. Here, it is important that no *tanh*-activation function follows, because otherwise the pendulums state values would be limited to the interval $[-1;1]$.


```julia
# NeuralFMU setup
numInputs = length(defaultFmu.modelDescription.inputValueReferences)
numOutputs = length(defaultFmu.modelDescription.outputValueReferences)

net = Chain(inputs -> fmiInputDoStepCSOutput(defaultFmu, tStep, inputs),
            Dense(numOutputs, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numOutputs))
```




    Chain(
      var"#15#16"(),
      Dense(2, 16, tanh),                   [90m# 48 parameters[39m
      Dense(16, 16, tanh),                  [90m# 272 parameters[39m
      Dense(16, 2),                         [90m# 34 parameters[39m
    )[90m                   # Total: 6 arrays, [39m354 parameters, 1.758 KiB.



#### Definition of the CS-NeuralFMU

The instantiation of the CS-NeuralFMU is done as a one-liner. The FMU `defaultFmu`, the structure of the network `net`, start `tStart` and end time `tStop`, and the time steps `tSave` for saving are specified.


```julia
csNeuralFmu = CS_NeuralFMU(defaultFmu, net, (tStart, tStop); saveat=tSave);
```

#### Plot before training

Here the state trajactory of the *extForceFmu* is recorded. Doesn't really look like a pendulum yet, but the system is random initialized by default. In the later plots, the effect of learning can be seen.


```julia
solutionBefore = csNeuralFmu(extForce, tStep)
Plots.plot(tSave, collect(data[1] for data in solutionBefore), label="acc CS-NeuralFMU", linewidth=2)
```




    
![svg](simple_hybrid_CS_files/simple_hybrid_CS_33_0.svg)
    



#### Training of the CS-NeuralFMU

For the training of the CS-NeuralFMU the parameters are extracted. The known ADAM optimizer for minimizing the gradient descent is used as further passing parameters. In addition, the previously defined loss and callback function, as well as the number of epochs are passed.


```julia
# train
paramsNet = Flux.params(csNeuralFmu)

optim = ADAM()
Flux.train!(lossSum, paramsNet, Iterators.repeated((), 300), optim; cb=callb)
```

    ┌ Info: Loss [1]: 1.13461
    └ @ Main In[13]:8
    ┌ Info: Loss [21]: 0.11918
    └ @ Main In[13]:8
    ┌ Info: Loss [41]: 0.07046
    └ @ Main In[13]:8
    ┌ Info: Loss [61]: 0.04412
    └ @ Main In[13]:8
    ┌ Info: Loss [81]: 0.03026
    └ @ Main In[13]:8
    ┌ Info: Loss [101]: 0.02166
    └ @ Main In[13]:8
    ┌ Info: Loss [121]: 0.01649
    └ @ Main In[13]:8
    ┌ Info: Loss [141]: 0.0134
    └ @ Main In[13]:8
    ┌ Info: Loss [161]: 0.01144
    └ @ Main In[13]:8
    ┌ Info: Loss [181]: 0.01001
    └ @ Main In[13]:8
    ┌ Info: Loss [201]: 0.00881
    └ @ Main In[13]:8
    ┌ Info: Loss [221]: 0.00774
    └ @ Main In[13]:8
    ┌ Info: Loss [241]: 0.00676
    └ @ Main In[13]:8
    ┌ Info: Loss [261]: 0.00587
    └ @ Main In[13]:8
    ┌ Info: Loss [281]: 0.00507
    └ @ Main In[13]:8


#### Comparison of the plots

Here three plots are compared with each other and only the acceleration of the mass is considered. The first plot represents the *defaultFMU*, the second represents the *referenceFMU* and the third plot represents the result after training the CS-NeuralFMU. 


```julia
# plot results mass.a
solutionAfter = csNeuralFmu(extForce, tStep)

fig = Plots.plot(xlabel="t [s]", ylabel="mass acceleration [m/s^2]", linewidth=2,
                 xtickfontsize=12, ytickfontsize=12,
                 xguidefontsize=12, yguidefontsize=12,
                 legendfontsize=8, legend=:topright)

accNeuralFmu = collect(data[1] for data in solutionAfter)

Plots.plot!(fig, tSave, accDefault, label="defaultFMU", linewidth=2)
Plots.plot!(fig, tSave, accReference, label="referenceFMU", linewidth=2)
Plots.plot!(fig, tSave, accNeuralFmu, label="CS-NeuralFMU (300 eps.)", linewidth=2)
fig 
```




    
![svg](simple_hybrid_CS_files/simple_hybrid_CS_37_0.svg)
    



Finally, the FMU is cleaned-up.


```julia
fmiUnload(defaultFmu)
```

### Summary

Based on the plots, it can be clearly seen that the CS-NeuralFMU model is able to learn the shift of the initial position. Even after only 300 runs, the curves overlap very much, so no further training with more runs is needed.

### Source

[1] Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

