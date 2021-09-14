
# FMIFlux.jl Documentation

## What is FMIFlux.jl?
FMIFlux.jl is a free-to-use software library for the Julia programming language, which offers the ability to setup NeuralFMUs: You can place FMUs (fmi-standard.org) simply inside any feed-forward NN topology and still keep the resulting hybrid model trainable with a standard AD training process.

## How can I install FMIFlux.jl?
1. open a Julia-Command-Window, activate your preferred environment
1. go to package manager using ```]``` and type ```add FMIFlux```
```
julia> ]

(v.1.5.4)> add FMIFlux
```

If you want to check that everything works correctly, you can run the tests bundled with FMIFlux.jl:
```
julia> using Pkg

julia> Pkg.test("FMIFlux")
```

Additionally, you can check the version of FMIFlux.jl that you have installed with the ```status``` command.
```
julia> ]
(v.1.5.4)> status FMIFlux
```

Throughout the rest of the tutorial we assume that you have installed the FMIFlux.jl package and have typed ```using FMIFlux``` which loads the package:

```
julia> using FMIFlux
```

## How the documentation is structured?
Having a high-level overview of how this documentation is structured will help you know where to look for certain things. The xxx main parts of the documentation are :
- The __Tutorials__ section explains all the necessary steps to work with the library.
- The __examples__ section gives insight in what is possible with this Library while using short and easily understandable code snippets
- The __library functions__ sections contains all the documentation to the functions provided by this library

## What is currently supported in FMIFlux.jl?
- building and training ME-NeuralFMUs with the default Flux-Front-End
- building and training CS-NeuralFMUs


## What is under development in FMIFlux.jl?
- different modes for sensitivity estimation
- documentation
- more examples

# FMIFlux.jl Index

```@index
```
