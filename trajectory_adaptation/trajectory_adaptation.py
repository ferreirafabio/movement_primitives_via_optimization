import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def is_pos_def(x):
  return np.all(np.linalg.eigvals(x) > 0)



def traj_adapt(traj_d, start, goal, norm):
  '''
  :param traj: (T, n)
  :param start: (n,)
  :param goal: (n,)
  :param norm: (T,T) --> assert positive definite
  :return: adapted trajectory (T,n)
  '''
  assert is_pos_def(norm), "norm is not positive definite"

  #traj = np.ndarray(shape=(T, dim))

  fun = lambda traj: ( (traj_d - traj).T.dot(norm) ).dot(traj_d - traj)
  cons = ({'type': 'eq', 'fun': lambda traj: traj[0] - start},
          {'type': 'eq', 'fun': lambda traj: traj[-1] - goal})

  return minimize(fun, x0=np.array([2]), method='SLSQP', constraints=cons)


def finite_dif_matrix(size):
  '''
   finite differencing matrix
  :param size: size of the quadratic matrix
  :return: numpy array -> shape=(size, size)
   '''
  return 2 * np.diag(np.ones([size])) + np.diag(-np.ones([size - 1]), k=1) + np.diag(-np.ones([size - 1]), k=-1)



T = 1000
dim = 2

""" generate 2d example trajectory """
x = np.linspace(start=0, stop=4, num=T)
y = 0.25 * x**2

traj_d = np.asarray([x,y]).T

start = [0,0]
goal = [3,3]
K = finite_dif_matrix(T)
M = np.transpose(K).dot(K)

print(traj_adapt(traj_d, start, goal, M))


#plt.plot(x, y, A)
#plt.show()
