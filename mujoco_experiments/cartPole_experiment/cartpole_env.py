import mujoco_py
import numpy as np
import xml.etree.ElementTree as xmlet


class CartPoleEnv:
    def __init__(self, filename='cartpole.xml'):
        self.model = mujoco_py.load_model_from_path(filename)
        self.sim = mujoco_py.MjSim(self.model)
        self.state = []
        self.state.append(self.sim.data.qpos[0])
        self.state.append(0)
        if self.sim.data.qpos[1] == 0:
            sign = 1
        else:
            sign = np.sign(self.sim.data.qpos[1])

        self.state.append(np.asscalar(np.remainder(self.sim.data.qpos[1], sign * 2 * np.pi)))
        self.state.append(0)
        self.dtime = 1/60

    # Reinforcement Learning Related Functions
    def step(self, action):
        done = False

        self.state[1] += action * self.dtime
        self.state[0] += self.state[1] * self.dtime


        state = []
        state.append(self.state[0])
        state.append(self.state[1])

        cost = 0
        if state[0] > 3 or state[0] < -3:
            done = True

        self.sim.data.qpos[0] = self.state[0]
        self.sim.forward()
        self.sim.step()

        if self.sim.data.qpos[1] == 0:
            sign = 1
        else:
            sign = np.sign(self.sim.data.qpos[1])

        pole_state = np.remainder(self.sim.data.qpos[1],  sign * 2 * np.pi)
        self.state[3] = (pole_state - self.state[2]) / self.dtime
        self.state[2] = pole_state

        cost += np.power(self.state[0],2) * 1.25 + np.power(self.state[1],2) * 1 + np.power(pole_state,2)*12 + np.power(self.state[3],2)*0.25 + 10*np.power(action,2)
        reward = cost
        state.append(self.state[2])
        state.append(self.state[3])

        return state, reward, done

    def reset(self, with_disturbance = False):
        self.sim = mujoco_py.MjSim(self.model)

        if with_disturbance is True:
            self.sim.data.qpos[0] += 1/60*100*np.random.randn()

        self.state = []
        self.state.append(self.sim.data.qpos[0])
        self.state.append(0)

        if self.sim.data.qpos[1] == 0:
            sign = 1
        else:
            sign = np.sign(self.sim.data.qpos[1])

        self.state.append(np.asscalar(np.remainder(self.sim.data.qpos[1], sign * 2 * np.pi)))
        self.state.append(0)
        self.dtime = 1/60

        state = []
        state.append(self.state[0])
        state.append(self.state[1])
        state.append(self.state[2])
        state.append(self.state[3])



        return state

    def create_viewer(self):
        self.viewer = mujoco_py.MjViewer(self.sim)

    def render(self):
        self.viewer.render()


if __name__ == '__main__':
    env = CartPoleEnv()
    while 1:
        state, reward, done = env.step(np.random.randn()*10)
        if done:
            break

        print('state is {}'.format(state))
        env.render()

