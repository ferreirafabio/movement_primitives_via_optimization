import matplotlib.pyplot as plt
import numpy as np
from trajectory_adaptation import adapt, get_finite_diff_matrix
from record_trajectory import record_single_trajectory, record_trajectories
from norm_learning import inner_minimization


def quadratic_trajectory_example():
  """ generate 2d example trajectory """
  T = 20

  t = np.linspace(start=0, stop=5, num=T)
  old_traj = 0.1* t ** 2 + 0.2 * t ** 3
  traj_d = np.asarray([t, old_traj]).T

  start = [0,0]
  goal = [5,25]

  K = get_finite_diff_matrix(T)
  M = np.transpose(K).dot(K)

  new_traj_y = adapt(traj_d[:, 1], start[1], goal[1], M)
  new_traj_x = adapt(traj_d[:, 0], start[0], goal[0], M)
  print(new_traj_y)


  plt.plot(t, old_traj)
  plt.plot(new_traj_x.x, new_traj_y.x)
  plt.plot([start[0], goal[0]], [start[1], goal[1]])
  plt.show()



def record_and_adapt():
  traj_d = record_trajectories(1)
  assert traj_d is not None, "no trajectory given"
  print(traj_d.shape)

  demonstrations = traj_d.shape[0]
  steps = traj_d.shape[1]

  for i in range(demonstrations):

    K = get_finite_diff_matrix(steps)
    M = np.transpose(K).dot(K)

    start = [traj_d[i,0,0]+0.2, traj_d[i,0,1]]
    goal = [traj_d[i,-1,0]+0.2, traj_d[i,-1,1]]

    new_traj_x = adapt(traj_d[i, :, 0], start[0], goal[0], M)
    new_traj_y = adapt(traj_d[i, :, 1], start[1], goal[1], M)

    plt.plot(traj_d[i, :, 0], traj_d[i, :, 1])
    plt.plot(new_traj_x.x, new_traj_y.x)
    plt.plot([start[0], goal[0]], [start[1], goal[1]])
    plt.show()


def record_and_do_inner_minimization():
  traj_d = record_trajectories(3)
  assert traj_d is not None, "no trajectory given"
  print(traj_d.shape)

  demonstrations = traj_d.shape[0]
  steps = traj_d.shape[1]


  # TODO: call inner minimization routine on trajectories
  for i in range(demonstrations):
    return NotImplementedError


record_and_do_inner_minimization()
#record_and_adapt()
#quadratic_trajectory_example()