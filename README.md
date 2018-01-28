# Movement Primitives via Optimization paper implementation
This repository describes a possible implementation of the paper "Movement Primitives via Optimization" (Dragan et al., 2015). It includes both trajectory adaptation with DMPs and norm learning via Lagrangian optimization.
Contributors: Jonas Rothfuss, Fabio Ferreira

## Trajectory adaptation with DMPs
We implemented the adaptation process (adaptation of a demonstrated trajectory to two new endpoints) in two ways:
1. As a Lagrangian minimization problem as specified in equation 2 (method used: scipy SLSQP). The adaptation assumes as a norm a finite difference matrix of a spring damper system (new positions are calculated based on accelerations).
2. As the optimization of the equation system implied by the Lagrangian optimization in equation 3 and 4 (method used: scipy broyden2). Again, a spring damper system is assumed.

## 
