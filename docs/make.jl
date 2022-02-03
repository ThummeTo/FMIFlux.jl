#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Documenter, FMIFlux

makedocs(sitename="FMIFlux.jl",
         format = Documenter.HTML(
            collapselevel = 1,
            sidebar_sitename = false
         ),
         pages= Any[
            "Introduction" => "index.md"
            "Examples" => [
                "Examples Overview" => "examples/overview.md"
                "Simple ME-NeuralFMU" => "examples/simple_hybrid_ME.md"
                "Simple CS-NeuralFMU" => "examples/simple_hybrid_CS.md"
            ]
            "Library Functions" => "library/overview.md"
            "Related Publication" => "related.md"
            "Contents" => "contents.md"
            ]
         )

deploydocs(repo = "github.com/ThummeTo/FMIFlux.jl.git", devbranch = "main")
