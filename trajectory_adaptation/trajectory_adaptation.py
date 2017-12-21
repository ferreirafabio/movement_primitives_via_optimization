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

  fun = lambda traj: ((traj_d - traj).T.dot(norm)).dot(traj_d - traj)
  cons = ({'type': 'eq', 'fun': lambda traj: traj[0] - start},
          {'type': 'eq', 'fun': lambda traj: traj[-1] - goal})

  # let's assume the trajectory must not be negative
  bnds=((0, None),)*T

  return minimize(fun, x0=np.zeros(shape=(T)), method='SLSQP', bounds=bnds, constraints=cons)



def finite_dif_matrix(size):
  '''
   finite differencing matrix
  :param size: size of the quadratic matrix
  :return: numpy array -> shape=(size, size)
   '''
  return 2 * np.diag(np.ones([size])) + np.diag(-np.ones([size - 1]), k=1) + np.diag(-np.ones([size - 1]), k=-1)



T = 100
dim = 2

""" generate 2d example trajectory """
x = np.linspace(start=0, stop=5, num=T)
old_traj = 0.25* x ** 2
traj_d = np.asarray([x, old_traj]).T

start = [0,0]
goal = [5,4.5]
K = finite_dif_matrix(T)
M = np.transpose(K).dot(K)

new_traj_y=traj_adapt(traj_d[:,1], start[1], goal[1], M)
new_traj_x=traj_adapt(traj_d[:,0], start[0], goal[0], M)
print(new_traj_x)
print(new_traj_y)

plt.plot(x, old_traj)
plt.plot(new_traj_x.x, new_traj_y.x)
plt.plot([start[0], goal[0]], [start[1], goal[1]])
plt.show()
