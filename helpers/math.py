import numpy as np
from scipy.linalg import polar

def is_pos_def(M):
  """ checks whether x^T * M * x > 0, M being the matrix to be checked
  :param M: the matrix to be checked
  :return: True if positive definite, False otherwise
  """
  return np.all(np.linalg.eigvals(M) > 0)


def loss_function(traj, traj_j):
  """
  indicator loss function for trajectories
  :param traj: (ndarray) first trajectory (the one to be minimized) of shape (n,)
  :param traj_j: (ndarray) second trajectory (from the demonstrations) of shape (n,)
  :return: 0 if trajectories are of the same shape and equal in terms of their elements, 1 otherwise
  """
  assert traj.shape == traj_j.shape
  if np.linalg.norm(traj - traj_j) < 10**-8:
    return 0
  return 1


def get_2nd_order_finite_diff_matrix(size):
  '''
  2nd order finite differencing matrix according to a spring damper system with which new positions are calculated based on
  the accelerations in a system.
  :param size: size of the quadratic matrix
  :return: the differencing matrix of shape (size, size)
   '''
  return 2 * np.diag(np.ones([size])) + np.diag(-np.ones([size - 1]), k=1) + np.diag(-np.ones([size - 1]), k=-1)

def get_1st_order_finite_diff_matrix(size):
  '''
  2nd order finite differencing matrix according to a spring damper system with which new positions are calculated based on
  the accelerations in a system.
  :param size: size of the quadratic matrix
  :return: the differencing matrix of shape (size, size)
   '''
  return np.diag(np.ones([size])) + np.diag(-np.ones([size - 1]), k=1)

# def project_norm_pos_def(M, eps=10**-8):
#   """
#   Projects a matrix M (norm) onto the cone of pos. (semi) def. matrices
#   :param M: a square matrix - numpy array of shape (m,m)
#   :return: P, the projection of M on the cone pos. semi-def. matrices
#   """
#   eigval, eigvec = np.linalg.eigh(M)
#   eigval_pos = np.maximum(eigval, eps)
#   P = eigvec.dot(np.diag(eigval_pos)).dot(eigvec.T)
#   assert P.shape == M.shape
#   return P


def project_norm_pos_def(A):
  """
  Calculates the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
  https://www.sciencedirect.com/science/article/pii/0024379588902236
  :param A: a square matrix
  :return A_pd: the projection of A onto the space pf positive definite matrices
  """
  assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"

  # symmetrize A into B
  B = (A + A.T) / 2

  # Compute the symmetric polar factor H of B
  _, H = polar(B)

  A_pd = (B + H) / 2

  # ensure symmetry
  A_pd = (A_pd + A_pd.T) / 2

  # test that A_pd is indeed PD. If not, then tweak it just a little bit
  pd = False
  k = 0
  while not pd:
    eig = np.linalg.eigvals(A_pd)
    pd = np.all(eig > 0)
    k += 1
    if not pd:
      mineig = min(eig)
      A_pd = A_pd + (-mineig * k ** 2 + 10**-8) * np.eye(A.shape[0])

  return A_pd



def ldl_decomp(A):
  """
  Computes the LDL decomposition of A
  :param A: symmetric matrix A
  :return: matrices (L, D) with same shape as A
  """
  import numpy as np
  A = np.matrix(A)
  if not (A.H == A).all():
    print("A must be Hermitian!")
    return None, None
  else:
    S = np.diag(np.diag(A))
    Sinv = np.diag(1 / np.diag(A))
    D = np.matrix(S.dot(S))
    Lch = np.linalg.cholesky(A)
    L = np.matrix(Lch.dot(Sinv))
    return L, D

def get_d_element(A, i):
  """
  computes the i-the diagonal element of the D matrix of the LDL decomposition of A
  :param A: symmetric Matrix
  :param i: integer denoting with
  :return:
  """
  assert i < A.shape[0]
  _, D = ldl_decomp(A)
  print(A)
  return np.diag(D)[i]