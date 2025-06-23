# Tests for Chebyshev and Fourier differentiation matrices at various orders.

## Issues
* I'm certain this is a consequence of limited precision but the higher-order derivatives(> 2), start to diverge after nx ~ 80
Not sure how to fix it. I have tried better ways to assemble the differentiation matrix but that doesn't help
Transforming the domain from [-1, 1] -> increases the error a little bit. Perhaps the rho reparametrization will improve it?
Status: Yes, rho parametrization improves the analytical high-order derivatives dramatically.

* Need to test derivatives when the Chebyshev grid is transformed. Tests completed!
Status: rho transformation will be a part of map coordinates and scale. General rho parametrization now possible!

* fourier\_diffmat1 agrees with fourier\_diffmat1\_alt but they are both wrong. The correct matrix is given by fourier\_diffmat
  need to check this further

* Perhaps add a minimizer that finds the correct value of scalars in the rho parametrization that reduces the errors throughout the derivatives. We need to solve this global minimization problem that find the best parmetrization ρ̂ = f(ρ) and then see how the error scales.
Status: Incomplete! This could be a feature in a future model where after a couple of solves the optimizer finds the right distribution of points f(ρ) which minimizes the error, making the solver adabptive

* Need to use the explicit second-order Fourier diffmatrix.

* The functions that calculate derivatives do not have the @jit flag. Ensure jittability before porting to GPUs
Status: Done!

* It would be better to ensure that the aspect ratio of any differential section is close to one => jacobian -> 1. It can be done by adding constant scaling factors of 2pi to the
theta and zeta directions.
