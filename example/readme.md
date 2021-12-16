# Example directory

## Getting Started

### Add Julia Kernel to Juypter
To run Julia as kernel in a jupyter notebook it is necessary to add the IJulia package.

1. Start the Julia REPL

    ```
    julia
    ```
 
 2. Add and build the IJulia package by typing inside the Julia REPL.

    ```julia
    using Pkg
    Pkg.add("IJulia")
    Pkg.build("IJulia")
    ```

3. Now you should be able to choose a Julia kernel in a Jupyter notebook.

