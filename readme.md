Why Second Order Networks? A Demonstration
=====

Explanation, Mark 1
----

Neural nets are nonlinear iterated function systems by construction. I tend to believe that the progression of the weights in weight space is a slice of another nonlinear iterated function system, also by construction. So I would tend to believe that the overall landscape of the optimization is suffused with directions with positive Lyapunov exponent, because if it's a fractal and an attractor, one considers it a strange attractor and begins suspecting that the dynamical process that creates it is chaotic. But that induces anisotropies in the optimization surface.

My current model in my head for how _both_ the optimization surface and the neural net function itself look like is the diffusion-limited aggregation. A diffusion-limited aggregation as you zoom in stays anisotropic in many directions at all scales. So our intuition about the behavior of optimization algorithms in each direction turns to something more like the Rosenbrock function, where the anisotropy defeats the normal gradient descent. But in all directions combined, it becomes more complicated.

Of course, folks are aware of this, and use 1.5-order optimization methods like ADAM. I believe that this is short measure and a true full damped second-order method must be used. I believe that not even Fisher information is sufficient (so Gauss-Newton is not sufficient, and Levenberg-Marquadt is not sufficient).

In the [other little project](https://github.com/howonlee/bobdobbsnewton), I think I have some progress (but not full progress) towards making full damped second order method practicable. But it is in some ways practicable now with more naive methods. It is this that is the subject of this demonstration, on CIFAR100 by just getting the whole damped inverse on a vastly smaller net.
