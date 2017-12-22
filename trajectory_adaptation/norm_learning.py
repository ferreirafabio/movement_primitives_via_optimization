import numpy as np
from scipy.optimize import minimize


def loss_function(traj, traj_j):
  if np.array_equal(traj, traj_j):
    return 0
  return 1



def inner_minimization(traj_i, traj_j, norm):

  fun = lambda traj: ((traj_i-traj).T.dot(norm).dot(traj_i-traj) - loss_function(traj, traj_j))

  cons = ({'type': 'eq', 'fun': lambda traj: traj[0] - traj_j[0]},
          {'type': 'eq', 'fun': lambda traj: traj[-1] - traj_j[1]})

  return minimize(fun, x0=np.ones(shape=(traj_i.shape[0])), method='SLSQP', bounds=None, constraints=cons,
                  tol=1e-17, options={'ftol': 1e-17, 'disp': True, 'maxiter': 20000})


