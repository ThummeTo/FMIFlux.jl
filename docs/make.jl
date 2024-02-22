#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Pkg; Pkg.develop(path=joinpath(@__DIR__,"../../FMIFlux.jl"))
using Documenter, FMIFlux
using Documenter: GitHubActions

makedocs(sitename="FMIFlux.jl",
        format = Documenter.HTML(
            collapselevel = 1,
            sidebar_sitename = false,
            edit_link = nothing,
            size_threshold_ignore = [joinpath("examples","juliacon_2023.md")]
        ),
        warnonly=true,
        pages= Any[
            "Introduction" => "index.md"
            "Examples" => [
                    "Overview" => "examples/overview.md"
                    "Simple CS-NeuralFMU" => "examples/simple_hybrid_CS.md"
                    "Simple ME-NeuralFMU" => "examples/simple_hybrid_ME.md"
                    "Growing Horizon ME-NeuralFMU" => "examples/growing_horizon_ME.md"
                    "JuliaCon 2023" => "examples/juliacon_2023.md"
                    "MDPI 2022" => "examples/mdpi_2022.md"
                    "Modelica Conference 2021" => "examples/modelica_conference_2021.md"
                    "Pluto Workshops" => "examples/workshops.md"
            ]
            "FAQ" => "faq.md"
            "Library Functions" => "library.md"
            "Related Publication" => "related.md"
            "Contents" => "contents.md"
            ]
        )

function deployConfig()
    github_repository = get(ENV, "GITHUB_REPOSITORY", "")
    github_event_name = get(ENV, "GITHUB_EVENT_NAME", "")
    if github_event_name == "workflow_run" || github_event_name == "repository_dispatch"
        github_event_name = "push"
    end
    github_ref = get(ENV, "GITHUB_REF", "")
    return GitHubActions(github_repository, github_event_name, github_ref)
end

deploydocs(repo = "github.com/ThummeTo/FMIFlux.jl.git", devbranch = "main", deploy_config = deployConfig())
