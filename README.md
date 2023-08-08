![FMIFlux.jl Logo](https://github.com/ThummeTo/FMIFlux.jl/blob/main/logo/dark/fmifluxjl_logo_640_320.png?raw=true "FMIFlux.jl Logo")
# FMIFlux.jl

## What is FMIFlux.jl?
[*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl) is a free-to-use software library for the Julia programming language, which offers the ability to simply place your FMU ([fmi-standard.org](http://fmi-standard.org/)) everywhere inside of your ML topologies and still keep the resulting models trainable with a standard (or custom) FluxML training process. This includes for example:
- NeuralODEs including FMUs, so called *Neural Functional Mock-up Units* (NeuralFMUs): 
You can place FMUs inside of your ML topology.
- PINNs including FMUs, so called *Functional Mock-Up Unit informed Neural Networks* (FMUINNs): 
You can evaluate FMUs inside of your loss function. 


[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ThummeTo.github.io/FMIFlux.jl/dev) 
[![Test (latest)](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/TestLatest.yml/badge.svg)](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/TestLatest.yml)
[![Test (LTS)](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/TestLTS.yml/badge.svg)](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/TestLTS.yml)
[![Examples](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Example.yml/badge.svg)](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Example.yml)
[![Build Docs](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Documentation.yml)
[![Run PkgEval](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Eval.yml/badge.svg)](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Eval.yml)
[![Coverage](https://codecov.io/gh/ThummeTo/FMIFlux.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ThummeTo/FMIFlux.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![FMIFlux Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/FMIFlux)](https://pkgs.genieframework.com?packages=FMIFlux)

## How can I use FMIFlux.jl?

1\. Open a Julia-REPL, switch to package mode using `]`, activate your preferred environment.

2\. Install  [*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl):
```julia-repl
(@v1) pkg> add FMIFlux
```

3\. If you want to check that everything works correctly, you can run the tests bundled with [*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl):
```julia-repl
(@v1) pkg> test FMIFlux
```

4\. Have a look inside the [examples folder](https://github.com/ThummeTo/FMIFlux.jl/tree/examples/examples) in the examples branch or the [examples section](https://thummeto.github.io/FMIFlux.jl/dev/examples/overview/) of the documentation. All examples are available as Julia-Script (*.jl*), Jupyter-Notebook (*.ipynb*) and Markdown (*.md*).

## What is currently supported in FMIFlux.jl?
- building and training ME-NeuralFMUs (NeuralODEs) with support for event-handling (*DiffEqCallbacks.jl*) and discontinuous sensitivity analysis (*SciMLSensitivity.jl*)
- building and training CS-NeuralFMUs 
- building and training NeuralFMUs consisiting of multiple FMUs
- building and training FMUINNs (PINNs)
- different AD-frameworks: ForwardDiff.jl (CI-tested), ReverseDiff.jl (CI-tested, default setting), FiniteDiff.jl (not CI-tested) and Zygote.jl (not CI-tested)
- ...

## What is under development in FMIFlux.jl?
- performance optimizations
- improved documentation
- more examples
- FMI3 integration
- ...

## What Platforms are supported?
[*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl) is tested (and testing) under Julia versions *v1.6* (LTS) and *v1* (latest) on Windows (latest) and Ubuntu (latest). MacOS should work, but untested.
[*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl) currently only works with FMI2-FMUs. 
All shipped examples are automatically tested under Julia version *v1* (latest) on Windows (latest).

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
