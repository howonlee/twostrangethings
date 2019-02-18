Two strange useless things to do with neural nets: a demonstration
=====

This is continuing on the work I did in the CSPAM thread ([first](https://github.com/howonlee/bobdobbshess), [second](https://github.com/howonlee/bobdobbsnewton)) and you'd probably be best off asking question or making comments there, although I'll probably end up hanging in the HN thread for a while. If you need a really friendly explanation, you are better off asking me questions in those places. I have a general inability to give understandable explanations except in reply to specific people.

But the demonstrations, at least, should be fairly approachable.

Unfriendly Explanation
----

Note: Download CIFAR100 from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and unpack it to the present directory to actually run these things. You also need numpy, but you shouldn't need anything else.

Neural nets are nonlinear iterated function systems by construction. I tend to believe that the progression of the weights in weight space is a slice of another nonlinear iterated function system, also by construction. So I would tend to believe that the overall landscape of the optimization is suffused with directions with positive Lyapunov exponent, because if it's a fractal and an attractor, one considers it a strange attractor and begins suspecting that the dynamical process that creates it is chaotic. But that induces anisotropies in the optimization surface.

My current poor-quality qualitative model in my head for how _both_ the optimization surface and the neural net function itself look like is the diffusion-limited aggregation. A diffusion-limited aggregation as you zoom in stays anisotropic in many directions at all scales. So our intuition about the behavior of optimization algorithms in each direction turns to something more like the Rosenbrock function, where the anisotropy defeats the normal gradient descent. But in all directions combined and at all scales, it should become more complicated.

I believe that this is actually unavoidable in information-processing systems that depend upon a manifold representation, because of the Margulis-Ruelle inequality(1), although I anticipate that there exist systems that avoid this by nontrivial constructions.

Of course, folks are aware of this, or at least the commonness of anisotropy, and use 1.5-order optimization methods like ADAM. I believe that this is short measure because of the multiple directions of the anisotropies and a very higher-order method, third or fourth or what-have-you method, must be used. I don't have that for you, but I have progress.

In the [other little project](https://github.com/howonlee/bobdobbsnewton), I think I have some progress (but not full progress) towards making second order methods practicable. But it is in some ways practicable now with more naive methods. It is this that is the subject of `slow_net.py`, on CIFAR100 by just getting the whole inverse on a vastly smaller net, with the SFN optimization of enforcing positive eigenspectrum of Hessian roughly following Dauphin et al (2) without a Krylov method. As I said, this is still short measure, and it isn't any better at CIFAR100. Also included is a demonstration of the fast second order method on a trivial network. `fast_net.py` is just an application of the fast FD newton's method on a trivial network (perceptron). Of course this is not any good either.

A very strange way to interpret things is pretty analogous to the nearest neighbor 1-lattice Ising model solution, taking the _layers_ as sites and the weights between layers as couplings. You need the second order because you need two terms in that Taylor expansion. Obviously nobody cares about the phase transition because it occurs in the unphysical T=0, but we would not care about unphysicalness. So a putative extensive transduction of the credit assignment and the extensive correlational structure at criticality in the second order phase transition would therefore be analogous. Of especial import in my mind is the Landau theory of phase transition in the Ising model, which needs the fourth order Taylor approximation (having symmetries which mean the Taylor approximation only has even terms). But many neural network things have been imputed to be critical and not much has come of those in many cases.

There are next steps to be taken in however so many weekends. I hypothesize that the third order and higher Taylor approximations would be straightforward because the third order tensor could be found in the expansion of Hv (the inverse, the expansion of H^{-1}v). I believe that the Pineda formulation of backpropagation could be a path to resolving the impasse I had previously with the circular definition of the second order backpropagation in nontrivial networks (3).

There is also an explanation in terms of literal credit assignment, but that warrants much longer discussion and I will talk of them later. Importantly, if we are executing that literal credit assignment, we can go as slowly as a pattern a week: in that case, 3rd and 4th order methods are practicable without a fast method, as long as we are willing to spend serious hard drive space.

1. Ruelle, D. (1978). An inequality for the entropy of differentiable maps. Bulletin of the Brazilian Mathematical Society, 9(1), 83-87.
2. Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., \& Bengio, Y. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. In Advances in neural information processing systems (pp. 2933-2941).
3. Pineda, F. (1987). Generalization of back-propagation to recurrent neural networks. Physical Review Letters,
19(59):2229–2232
