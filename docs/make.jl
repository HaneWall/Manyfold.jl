using Documenter, DocumenterVitepress

using Manyfold

makedocs(;
    clean = true,
    modules=[Manyfold],
    authors="Hannes Wallner",
    repo="https://github.com/HaneWall/Manyfold.jl",
    sitename="Manyfold.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/HaneWall/Manyfold.jl",
        devurl = "dev",
        deploy_url = "HaneWall.github.io/Manyfold.jl",
    ),
    pages=[
        "Overview" => "index.md",
        "Diffusion Maps" => [
              "diffmaps/getting-started.md",
        ],
    ],
    warnonly = true,
)

deploydocs(;
    repo="github.com/HaneWall/Manyfold.jl",
    push_preview=true,
)
