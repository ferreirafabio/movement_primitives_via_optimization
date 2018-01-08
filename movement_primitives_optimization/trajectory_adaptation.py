import numpy as np
from scipy.optimize import minimize, newton_krylov, broyden2
from movement_primitives_optimization.helpers import math

def optimize(traj_d, start, goal, norm):
  '''
  this function minimizes the Lagrangian as specified in "Movement Primitives via Optimization" (Dragan
  2015) in equation 2 in order to adapt a demonstrated trajectory to two new endpoints (start and goal). The adaptation
  assumes as a norm a finite difference matrix of a spring damper system (new positions are calculated based on
  accelerations).
  :param traj: (T, n)
  :param start: (n,)
  :param goal: (n,)
  :param norm: (T,T) --> assert positive definite
  :return: adapted trajectory (T,n)
  '''
  assert math.is_pos_def(norm), "norm must be positive definite"

  fun = lambda traj: ((traj_d - traj).T.dot(norm)).dot(traj_d - traj)
  cons = ({'type': 'eq', 'fun': lambda traj: traj[0] - start},
          {'type': 'eq', 'fun': lambda traj: traj[-1] - goal})


  return minimize(fun, x0=traj_d, method='SLSQP', bounds=None, constraints=cons,
   tol=1e-17, options={'ftol': 1e-17, 'disp': True, 'maxiter': 20000}).x

def optimize_via_equation_system(traj_d, start, goal, norm):
  '''
    this function solves the equation system implied by the Lagrangian Optimization in "Movement Primitives via Optimization" (Dragan
    2015) in equation 3 and 4 in order to adapt a demonstrated trajectory to two new endpoints (start and goal). The adaptation
    assumes as a norm a finite difference matrix of a spring damper system (new positions are calculated based on
    accelerations).
    :param traj: (T, n)
    :param start: (n,)
    :param goal: (n,)
    :param norm: (T,T) --> assert positive definite
    :return: adapted trajectory (T,n)
    '''
  assert math.is_pos_def(norm), "norm must be positive definite"
  traj_len = traj_d.shape[0]

  start_goal_vec = np.zeros(traj_len)
  start_goal_vec[0] = start
  start_goal_vec[-1] = goal

  b = norm.dot(traj_d - start_goal_vec)

  mask1 = np.zeros(traj_len)
  mask1[0], mask1[-1] = 1, 1

  mask2 = np.ones(traj_len)
  mask2[0], mask2[-1] = 0, 0

  fun = lambda x: norm.dot(np.multiply(mask2, x)) - np.multiply(mask1, x) - b

  traj = broyden2(fun, traj_d)
  traj[0] = start
  traj[-1] = goal
  return traj



def adapt_all_dimensions(traj_d, start, goal, method="SQP"):
  """
  This function adapts a demonstrated trajectory to a given start and goal endpoint as specified in "Movement
  Primitives via Optimization" (Dragan 2015). The adaptation assumes as a norm a finite difference matrix of a
  spring damper system (new positions are calculated based on accelerations).
  :param traj_d: (ndarray) The trajectory of shape: (length/time steps of trajectory, dimensions)
  :param start: (ndarray) 1st equality constraint, the new start point, shape: (dimensions,)
  :param goal: (ndarray) 2nd equality constraint the new goal point, shape: (dimensions,)
  :param method: method of optimization (SQP - sequential quadaratic programming, EQ - solve equation implied by Langrangian optimization)

  :return: The adapted trajectory as an ndarray of shape [n_dim, len_traj], with n_dim being the number of
  trajectories and len_traj being the length or number of time steps of a trajectory. The order of the dimensions is
  the same as the order in the input.
  """
  assert method in ["SQP", "EQ"]

  M = math.get_finite_diff_matrix(traj_d.shape[0])

  dimensions = traj_d.shape[1]
  new_trajectories = []

  for dim in range(dimensions):
    if method is "SQP":
      new_traj = optimize(traj_d[:, dim], start[dim], goal[dim], M)
    else: # EQ
      new_traj = optimize_via_equation_system(traj_d[:, dim], start[dim], goal[dim], M)
    new_trajectories.append(new_traj)

  return np.asarray(new_trajectories)




