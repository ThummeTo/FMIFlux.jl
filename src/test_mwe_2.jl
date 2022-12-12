using ForwardDiff
using DifferentialEquations

function lotka_volterra!(dx,x,p,t)
  dx[1] = x[1] * sum(p)
end

p = ones(20000); 
u0 = [1.0]
prob = ODEProblem(lotka_volterra!,u0,(0.0,10.0),p)

function f(x)
    _prob = remake(prob,u0=x[1:2],p=x[3:end])
    sol = solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=1)[1,:]
    return sol[1].value
end

x = [u0;p]

print("success before")

grad_conf = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{600}());
grad = ForwardDiff.gradient(f, x, grad_conf);

print("success after")