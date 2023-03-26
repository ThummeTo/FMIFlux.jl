using PkgEval
using FMIFlux

config = Configuration(; julia="1.8");

package = Package(; name="FMIFlux");

@info "PkgEval"
result = evaluate([config], [package])

@info "Result"
println(result)

@info "Log"
println(result.log)