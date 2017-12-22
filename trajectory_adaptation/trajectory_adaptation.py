import numpy as np
from scipy.optimize import minimize


def is_pos_def(x):
  return np.all(np.linalg.eigvals(x) > 0)


def adapt(traj_d, start, goal, norm):
  '''
  :param traj: (T, n)
  :param start: (n,)
  :param goal: (n,)
  :param norm: (T,T) --> assert positive definite
  :return: adapted trajectory (T,n)
  '''
  assert is_pos_def(norm), "norm is not positive definite"

  fun = lambda traj: ((traj_d - traj).T.dot(norm)).dot(traj_d - traj)
  cons = ({'type': 'eq', 'fun': lambda traj: traj[0] - start},
          {'type': 'eq', 'fun': lambda traj: traj[-1] - goal})

  # let's assume the trajectory must not be negative
  #bnds=((0, None),)*traj_d.shape[0]

  return minimize(fun, x0=np.ones(shape=(traj_d.shape[0]))*-1, method='SLSQP', bounds=None, constraints=cons,
   tol=1e-17, options={'ftol': 1e-17, 'disp': True, 'maxiter': 20000})



def get_finite_diff_matrix(size):
  '''
   finite differencing matrix
  :param size: size of the quadratic matrix
  :return: numpy array -> shape=(size, size)
   '''
  return 2 * np.diag(np.ones([size])) + np.diag(-np.ones([size - 1]), k=1) + np.diag(-np.ones([size - 1]), k=-1)



