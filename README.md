![FMIFlux.jl Logo](https://github.com/ThummeTo/FMIFlux.jl/blob/main/logo/dark/fmifluxjl_logo_640_320.png?raw=true "FMIFlux.jl Logo")
# FMIFlux.jl

## What is FMIFlux.jl?
[*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl) is a free-to-use software library for the Julia programming language, which offers the ability to set up NeuralFMUs just like NeuralODEs: You can place FMUs ([fmi-standard.org](http://fmi-standard.org/)) simply inside any feed-forward ANN topology and still keep the resulting hybrid model trainable with a standard AD training process.

[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ThummeTo.github.io/FMIFlux.jl/dev) 
[![Run Tests](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Test.yml/badge.svg)](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Test.yml)
[![Run Examples](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Example.yml/badge.svg)](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Example.yml)
[![Build Docs](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Documentation.yml)
[![Coverage](https://codecov.io/gh/ThummeTo/FMIFlux.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ThummeTo/FMIFlux.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)


## How can I use FMIFlux.jl?

1\. Open a Julia-REPL, switch to package mode using `]`, activate your preferred environment.

2\. Install  [*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl):
```julia-repl
(@v1.6) pkg> add FMIFlux
```

3\. If you want to check that everything works correctly, you can run the tests bundled with [*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl):
```julia-repl
(@v1.6) pkg> test FMIFlux
```

4\. Have a look inside the [examples folder](https://github.com/ThummeTo/FMIFlux.jl/tree/examples/examples) in the examples branch or the [examples section](https://thummeto.github.io/FMIFlux.jl/dev/examples/overview/) of the documentation. All examples are available as Julia-Script (*.jl*), Jupyter-Notebook (*.ipynb*) and Markdown (*.md*).

## What is currently supported in FMIFlux.jl?
- building and training ME-NeuralFMUs (event-handling is BETA) with the default Flux-Front-End
- building and training CS-NeuralFMUs with the default Flux-Front-End
- ...

## What is under development in FMIFlux.jl?
- performance optimizations
- different modes for sensitivity estimation
- improved documentation
- explicitely time-dependent systems
- more examples
- ...

## What Platforms are supported?
[*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl) is tested (and testing) under Julia versions *1.6 LTS* and *latest* on Windows (latest) and Ubuntu (latest). MacOS should work, but untested.

## What FMI.jl-Library should I use?
![FMI.jl Family](https://github.com/ThummeTo/FMI.jl/blob/main/docs/src/assets/FMI_JL_family.png?raw=true "FMI.jl Family")
To keep dependencies nice and clean, the original package [*FMI.jl*](https://github.com/ThummeTo/FMI.jl) had been split into new packages:
- [*FMI.jl*](https://github.com/ThummeTo/FMI.jl): High level loading, manipulating, saving or building entire FMUs from scratch
- [*FMIImport.jl*](https://github.com/ThummeTo/FMIImport.jl): Importing FMUs into Julia
- [*FMIExport.jl*](https://github.com/ThummeTo/FMIExport.jl): Exporting stand-alone FMUs from Julia Code
- [*FMICore.jl*](https://github.com/ThummeTo/FMICore.jl): C-code wrapper for the FMI-standard
- [*FMIBuild.jl*](https://github.com/ThummeTo/FMIBuild.jl): Compiler/Compilation dependencies for FMIExport.jl
- [*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl): Machine Learning with FMUs (differentiation over FMUs)
- [*FMIZoo.jl*](https://github.com/ThummeTo/FMIZoo.jl): A collection of testing and example FMUs

## How to cite?
Tobias Thummerer, Johannes Stoljar and Lars Mikelsons. 2022. **NeuralFMU: presenting a workflow for integrating hybrid NeuralODEs into real-world applications.** Electronics 11, 19, 3202. [DOI: 10.3390/electronics11193202](https://doi.org/10.3390/electronics11193202)

Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

## Related publications?
Tobias Thummerer, Johannes Tintenherr, Lars Mikelsons 2021. **Hybrid modeling of the human cardiovascular system using NeuralFMUs** Journal of Physics: Conference Series 2090, 1, 012155. [DOI: 10.1088/1742-6596/2090/1/012155](https://doi.org/10.1088/1742-6596/2090/1/012155)