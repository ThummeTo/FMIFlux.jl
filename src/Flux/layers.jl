#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMIFlux.FMIImport: FMU2

Flux.@layer FMU2
Flux.@layer FMUParameterRegistrator
Flux.@layer FMUTimeLayer
Flux.@layer ParameterRegistrator
Flux.@layer SimultaniousZeroCrossing
Flux.@layer ShiftScale
Flux.@layer ScaleShift
Flux.@layer ScaleSum
Flux.@layer CacheLayer
Flux.@layer CacheRetrieveLayer

Flux.trainable(a::FMU2) = (;)
Flux.trainable(a::FMUParameterRegistrator) = (; p = a.p)
Flux.trainable(a::FMUTimeLayer) = (; offset = a.offset)
Flux.trainable(a::ParameterRegistrator) = (; p = a.p)
Flux.trainable(a::SimultaniousZeroCrossing) = (; m = a.m)
Flux.trainable(a::ShiftScale) = (; shift = a.shift, scale = a.scale)
Flux.trainable(a::ScaleShift) = (; scale = a.scale, shift = a.shift)
Flux.trainable(a::ScaleSum) = (; scale = a.scale)
Flux.trainable(a::CacheLayer) = (;)
Flux.trainable(a::CacheRetrieveLayer) = (;)

Flux.Functors.@functor FMU2 () # prevent FMUs from being parsed by Flux
Flux.Functors.@functor FMUParameterRegistrator (p,)
Flux.Functors.@functor FMUTimeLayer (offset,)
Flux.Functors.@functor ParameterRegistrator (p,)
Flux.Functors.@functor SimultaniousZeroCrossing (m,)
Flux.Functors.@functor ShiftScale (shift, scale)
Flux.Functors.@functor ScaleShift (scale, shift)
Flux.Functors.@functor ScaleSum (scale,)
Flux.Functors.@functor CacheLayer ()
Flux.Functors.@functor CacheRetrieveLayer ()

# old Flux
# Flux.@functor FMUParameterRegistrator (p,)
# Flux.@functor FMUTimeLayer (offset,)
# Flux.@functor ParameterRegistrator (p,)
# Flux.@functor SimultaniousZeroCrossing (m,)
# Flux.@functor ShiftScale (shift, scale)
# Flux.@functor ScaleShift (scale, shift)
# Flux.@functor ScaleSum (scale,)
