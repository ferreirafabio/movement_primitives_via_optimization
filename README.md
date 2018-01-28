# Implementation of the paper "Movement Primitives via Optimization" (Dragan et al., 2015)
This repository describes a possible implementation of the paper "Movement Primitives via Optimization" (Dragan et al., 2015) [1]. It includes both trajectory adaptation with DMPs and norm learning via Lagrangian optimization.


## Trajectory adaptation with DMPs
We implemented the adaptation process for demonstrated trajectories in two different ways:
1. As a Lagrangian minimization problem as specified in equation 2 (method used: scipy SLSQP). The adaptation assumes as a norm a finite difference matrix of a spring damper system (new positions are calculated based on accelerations).
2. As the optimization of the equation system implied by the Lagrangian optimization in equation 3 and 4 (method used: scipy broyden2). Again, a spring damper system is assumed.

## Norm learning
Given that the user provided sample adaptations in addition to the demonstrations, the norm learning part is executed by applying Maximum Margin Planning. The functions iteratively apply the following three steps,
  given pairs of trajectories (traj_i, traj_j) in DxD (D being the set of user demonstrations):
    1) compute the optimal solution to the "inner minimization problem" (right term in eq. 19)
    2) compute the gradient update for the norm with a hyper-parameter alpha, update the norm
    3) project the updated norm to the space of pos. def. matrices, unless "iterations" not exceeded, go to 1)
    Note: the projection is carried out by computing the nearest symmetric pos. def. matrix in the Frobenius norm according to [2], a method based on the polar decomposition. A second method we examined is based on the Cholesky decomposition.
    

### References
* [1] "Movement Primitives via Optimization", Dragan et al., 2015, https://www.ri.cmu.edu/pub_files/2015/5/DMP_IEEE.pdf, http://ieeexplore.ieee.org/document/7139510/, 
* [2] "Computing a nearest symmetric positive semidefinite matrix", Higham, 1988, https://www.sciencedirect.com/science/article/pii/0024379588902236


### Credits
Thank you to Drew Bagnell and Anca Dragan for giving support via e-mail.
_Contributors: Jonas Rothfuss, Fabio Ferreira_
