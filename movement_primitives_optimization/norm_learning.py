import numpy as np
from scipy.optimize import minimize
import pandas as pd
from movement_primitives_optimization.helpers import math
import itertools


def inner_minimization(traj_i, traj_j, M):
  """
  Applies the right term of eq. 19 in "Movement Primitives via Optimization" (Dragan et al., 2015) via Lagrangian
  optimization (SLSQP method with constraints as specified in the paper). Each dimension is optimized separately and
  their min-values compose a new vector of shape (# dimensions,).
  :param traj_i: First trajectory of shape (time steps of trajectory, dimensions)
  :param traj_j: Second trajectory of shape (time steps of trajectory, dimensions)
  :param norm: A norm under which the optimization process is executed.
  :return: A vector of shape (# dimensions,) that is composed of the min-values of each separate dimension
  Lagrangian-optimization
  """
  assert traj_i.shape == traj_j.shape
  assert traj_i.ndim == 1, traj_j.ndim == 1

  fun = lambda traj: (traj_i - traj).T.dot(M).dot(traj_i - traj) - math.loss_function(traj, traj_j)

  cons = ({'type': 'eq', 'fun': lambda traj: traj[0] - traj_j[0]},
          {'type': 'eq', 'fun': lambda traj: traj[-1] - traj_j[-1]})

  init_guess = traj_j + np.random.normal(size=(traj_i.shape[0]), scale=0.01)

  opt_result = minimize(fun, x0=init_guess, method='SLSQP', constraints=cons,
                        options={'maxiter': 20000, "disp": False})

  return opt_result.x, opt_result.fun


def margin_loss(demonstrations, M):
  ndim_traj = demonstrations[0].shape[1]

  loss = 0
  for traj_i, traj_j, dim in itertools.product(demonstrations, demonstrations, range(ndim_traj)):
    _, inner_min_result = inner_minimization(traj_i[:,dim], traj_j[:,dim], M)
    loss += (traj_i[:,dim]-traj_j[:,dim]).T.dot(M).dot(traj_i[:,dim]-traj_j[:,dim]) - inner_min_result

  return loss


def learn_norm_via_opt(demonstrations, init_norm):
  fun = lambda K: margin_loss(demonstrations, K.T.dot(K))

  print("init_norm shape", init_norm.shape[0])

  #cons = [{'type': 'ineq', 'fun': lambda K: - math.get_d_element(K.T.dot(K),i)} for i in range(init_norm.shape[0])]


  opt_result = minimize(fun, x0=init_norm,
                        options={'maxiter': 20000, "disp": False})

  return opt_result.x

def learn_norm(demonstrations, init_norm, alpha=0.01, iterations=1000):
  """
  Implementation of norm learning from the paper "Movement Primitives via Optimization" (Dragan et al., 2015)
  Specifically, this function learns a norm given that the user provides not only demonstrations but also adaptations
  by applying Maximum Margin Planning. The function iteratively applies the following three steps,
  given pairs of trajectories (traj_i, traj_j) \in DxD (D being the set of user demonstrations):
    1) compute the optimal solution to the "inner minimization problem" (right term in eq. 19)
    2) compute the gradient update for the norm with a hyper-parameter alpha, update the norm
    3) project the updated norm to the space of pos. def. matrices, repeat
  :param demonstrations: the trajectories, can be a pandas DataFrame or a list of ndarrays with shape (time steps,
  dimensions)
  :param init_norm: the initial norm from where we the norm updates start from
  :param alpha: learning rate for the norm update
  :param iterations: number of iterations the norm should be updates
  :return: the learned norm of the same shape as init_norm
  """
  assert demonstrations, "no trajectory given"
  assert alpha > 0
  assert math.is_pos_def(init_norm)

  ndim_traj = demonstrations[0].shape[1]

  if isinstance(demonstrations, pd.DataFrame):
    # flatten required to convert 2d array to 1d
    demonstrations = demonstrations.values.flatten()


  M = init_norm


  def calculate_gradients(traj_i, traj_j, dim):
    traj_ij, _ = inner_minimization(traj_i[:, dim], traj_j[:, dim], M)
    grad = (traj_i[:, dim] - traj_j[:, dim]).dot((traj_i[:, dim] - traj_j[:, dim]).T) - (traj_i[:, dim] - traj_ij).dot(
      (traj_i[:, dim] - traj_ij).T)
    grads.append(grad)

  for k in range(iterations):
    grads = []
    #Parallel(n_jobs=NUM_CORES)(delayed(calculate_gradients)(traj_i, traj_j, dim)
    #                           for traj_i, traj_j, dim in itertools.product(demonstrations, demonstrations, range(ndim_traj)))
    for traj_i, traj_j, dim in itertools.product(demonstrations, demonstrations, range(ndim_traj)):
      calculate_gradients(traj_i, traj_j, dim)

    mean_grad = np.mean(grads)
    M -= alpha * mean_grad

    M = math.project_norm_pos_def(M)
    print("LOSS :", margin_loss(demonstrations, M))

  return M


