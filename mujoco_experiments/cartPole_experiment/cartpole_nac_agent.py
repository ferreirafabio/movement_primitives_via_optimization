import random
import gym
import math
import numpy as np
from numpy.linalg import inv, norm, pinv
import cartpole_env

class SimpleNACAgent():
    def __init__(self, num_eps = 2000, n_win_ticks=195, lam=1.0, gamma = 1, alpha = 0.01, beta = 0.7, errangle = np.pi/2, theta = np.array(np.random.randn(4)),quiet=False):
        self.env = cartpole_env.CartPoleEnv()
        self.action_set = list([0,1])
        self.theta = theta
        self.sigma = 0.1
        self.gamma = gamma
        self.lam = lam
        self.num_eps = num_eps
        self.n_win_ticks = n_win_ticks
        self.beta = beta
        self.alpha = alpha
        self.errangle = errangle


    def run_policy(self, state):
        u = self.theta.dot(state)
        u_s = random.gauss(u, self.sigma)
        return u_s

    def deriv_policy(self, state, action):
        mu = self.theta.dot(state)
        # gauss = 1 / np.sqrt(2*np.pi * self.sigma * self.sigma) * np.exp(-np.power(action - mu, 2.) / (2 * np.power(self.sigma, 2.)))
        deriv_vec = - action / np.power(self.sigma,2) * state
        return deriv_vec

    def phi(self, state):
        return state.copy()

    def train(self):

        At = 0
        bt = 0
        zt = 0

        max_iter = 0
        all_total_reward = 0
        for i in range(self.num_eps):
            x = np.array(self.env.reset(with_disturbance=True))


            if np.remainder(i+1, 100) == 0:
                atr = all_total_reward / 100
                print('i: {} , average total reward is {}'.format(i,atr))
                all_total_reward = 0

            done = False
            total_reward = 0
            nums = 0
            phi_A = []
            b = []
            while not done and nums < 100:
                nums += 1
                u = self.run_policy(x)
                x_1, reward, done = self.env.step(u)
                reward = reward
                x_1 = np.array(x_1)
                total_reward += reward

                phi_tilde = [0, 0, 0, 0]
                derivp = self.deriv_policy(x, u)
                phi_hat = [derivp[0], derivp[1], derivp[2], derivp[3]]

                phi_x = self.phi(x)
                phi_x_1 = self.phi(x_1)
                for j in range(len(phi_x)):
                    phi_tilde.append(phi_x_1[j])
                    phi_hat.append(phi_x[j])

                # print('x is {}'.format(x))
                # print('phi_x is {}'.format(phi_x))

                # phi_tilde = [x_1[0], x_1[1], x_1[2], x_1[3],0, 0, 0, 0]
                # derivp = self.deriv_policy(x, u)
                # phi_hat = [x[0], x[1], x[2], x[3],derivp[0], derivp[1], derivp[2], derivp[3]]
                phi_tilde = np.matrix(phi_tilde).transpose()
                phi_hat = np.matrix(phi_hat).transpose()
                zt = self.lam * zt + phi_hat
                At = At + np.matmul(zt, (phi_hat - self.gamma * phi_tilde).transpose())
                bt = bt + np.multiply(zt, reward)
                pinvAt = np.linalg.pinv(At)
                para_vec = np.matmul(pinvAt, bt)
                wt = np.array([np.asscalar(para_vec[0][0]),
                      np.asscalar(para_vec[1][0]),
                      np.asscalar(para_vec[2][0]),
                      np.asscalar(para_vec[3][0])])
                if i == 0:
                    angle = 1
                    w0 = wt
                else:
                    angle = math.acos(w0.dot(wt) / (norm(w0) * norm(wt) + 1e-5))

                if angle < self.errangle:
                    self.theta = self.theta + np.array(wt) * self.alpha
                    zt = self.beta * zt
                    At = self.beta * At
                    bt = self.beta * bt
                    x = x_1
                    w0 = wt

            #     cphi_A = np.array(phi_hat) - self.gamma * np.array(phi_tilde)
            #     phi_A.append(cphi_A)
            #     b.append(reward)
            #
            # A = np.matrix(phi_A)
            # pinvA = np.linalg.pinv(A)
            # pvec = pinvA * np.matrix(b).transpose()
            # w1 = pvec.A1[0:4]
            # if i == 0:
            #     angle = 1
            #     w0 = w1
            # else:
            #     angle = np.arccos(w0.dot(w1) / (norm(w0) * norm(w1) + 1e-5))
            #
            # self.theta += w1 * self.alpha


            if nums > max_iter:
                max_iter = nums
            all_total_reward += total_reward
            #print('total reward is {}'.format(total_reward))

        return self.theta


    def test(self):
        state = np.array(self.env.reset()).transpose()
        self.env.create_viewer()
        total_reward = 0
        for e in range(10000):
            self.env.render()
            action = self.run_policy(state)
            print('action is {}'.format(action))
            next_state, reward, done = self.env.step(action)
            state = np.array(next_state).transpose()
            total_reward += reward
            if done:
                print('Episode {} Done ... '.format(e))
                return total_reward


if __name__ == '__main__':
    # theta = np.array([-83.27341047, 15.8037482,  -217.6333067, 28.96246042])
    theta = np.array([-0.27584952, 0.69535139, -29.44710623,  13.30643324])
    agent = SimpleNACAgent(theta = theta)
    # theta = agent.train()

    print('theta is {}'.format(theta))
    reward = agent.test()
    print('cost is {}'.format(reward))



