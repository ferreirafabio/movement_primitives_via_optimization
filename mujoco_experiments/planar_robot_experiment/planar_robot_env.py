import mujoco_py
import numpy as np
import xml.etree.ElementTree as xmlet


class PlanarRobotEnv:
    def __init__(self, filename='planar_robot.xml', obj_pose='', target_pose='', generate_motion=False):
        self.armlen = 0.6
        self.handlen = 0.4
        self.grasp_joint = 0.25

        self.generate_motion = generate_motion
        if generate_motion is True:
            print('Note: if obj_pose and target_pose are set, they will be ignored !!!')
            self.qTraj, obj_pose_vec, target_pose_vec = self.generate_rand_pick_place_movement(dtime=0.02)
            obj_pose = ' '.join(str(x) for x in obj_pose_vec)
            obj_pose = obj_pose + ' ' + str(0)
            target_pose = ' '.join(str(x) for x in target_pose_vec)
            target_pose = target_pose + ' ' + str(-0.1)
            print('obj_pose is {}'.format(obj_pose))
            print('target_pose is {}'.format(target_pose))

        tree = xmlet.parse(filename)
        if obj_pose != '':
            xmlreader = tree.getroot()
            for body in xmlreader.iter('body'):
                if 'name' in body.attrib and body.attrib['name'] == 'targetObj':
                    body.attrib['pos'] = obj_pose

        if target_pose != '':
            xmlreader = tree.getroot()
            for body in xmlreader.iter('geom'):
                if 'name' in body.attrib and body.attrib['name'] == 'target_pose':
                    body.attrib['pos'] = target_pose

        tree.write('planar_robot_tmp.xml')
        self.model = mujoco_py.load_model_from_path('planar_robot_tmp.xml')
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.current_state =[]
        self.current_state.append([self.sim.data.qpos[i] for i in range(3)])
        self.current_state.append(self.sim.data.get_body_xpos('targetObj'))


    # Planar Robot Model
    def transmat(self, transl, theta):
        rotm = np.matrix([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        tralm = np.matrix([[1,0,transl], [0,1,0], [0,0,1]])
        return np.matmul(rotm, tralm)

    def fkine(self, qpos):
        tra1 = self.transmat(self.armlen, qpos[0])
        tra2 = self.transmat(self.armlen, qpos[1])
        tra3 = self.transmat(self.handlen/2, 0)
        veca = np.matrix([[0],[0],[1]])
        pa1 = np.matmul(tra1, veca)
        pa2 = np.matmul(np.matmul(tra1, tra2), veca)

        ctcp = np.matmul(np.matmul(np.matmul(tra1, tra2),tra3), veca)
        ovec = ctcp[0:2] - pa2[0:2]
        ovec = np.divide(ovec, np.linalg.norm(ovec))

        if np.sign(ovec[1]) != 0:
            origSign = np.sign(ovec[1])
        else:
            origSign = 1

        alg = origSign * np.arccos(np.dot(ovec.A1,np.array([1,0])))
        posxy = ctcp[0:2].A1
        pos = np.array([posxy[0], posxy[1], alg])

        return pos

    def Jacobian(self, qpos):
        L0 = self.armlen
        L1 = self.armlen

        q0 = qpos[0]
        q1 = qpos[1]

        jMat = np.matrix([[-L0 * np.sin(q0) - L1 * np.sin(q0 + q1), -L1 * np.sin(q0 + q1)],
                          [L0 * np.cos(q0) + L1 * np.cos(q0 + q1),  L1 * np.cos(q0 + q1)],
                          [1, 1]])
        return jMat

    def ikine(self, tcp, qpos, max_Iter=10000, err_tolerance=1e-3):
        itr = 0
        ctcp = self.fkine(qpos)
        cq = qpos.copy()
        t = 0.01
        c_error = np.linalg.norm(tcp[0:2] - ctcp[0:2])
        p_error = 0

        print('error before IK is {}'.format(c_error))
        print('cq before IK is {}'.format(cq))
        while c_error > err_tolerance and np.absolute(c_error - p_error) > 1e-3 and itr < max_Iter:
            itr += 1
            jMat = self.Jacobian(cq)
            jMatInv = np.linalg.pinv(jMat)
            tcp_vel = (tcp - ctcp)/t
            q_vel = np.matmul(jMatInv, np.asmatrix(tcp_vel).transpose())
            cq[0:2] = cq[0:2] + q_vel.A1 * t
            ctcp = self.fkine(cq)
            p_error = c_error
            c_error = np.linalg.norm(tcp[0:2] - ctcp[0:2])

        print('error is {}'.format(c_error))
        print('cq after IK is {}'.format(cq))
        return cq

    def generate_rand_pick_place_movement(self, dtime=1, err_tol=1e-2):
        kp = 0.1
        kd = kp * kp / 4

        qpoint = []
        qpoint.append(np.array([0, 0, 0]))
        qobj_1 = np.pi / 2 + np.random.random() * (np.pi - np.pi / 2)
        qobj_2 = np.pi / 4 + np.random.random() * np.pi / 4
        qobj_3 = -self.grasp_joint
        qobj = np.array([qobj_1, qobj_2, qobj_3])

        offset = 0.7
        tobj = self.fkine(qobj)
        tbefore = tobj.copy()
        tbefore[0:2] -= np.array([np.cos(tobj[2]), np.sin(tobj[2])]) * offset
        print('tobj is {}'.format(tobj))
        print('tbefore is {}'.format(tbefore))

        obj_pose = tobj[0:2] + np.array([np.cos(tobj[2]), np.sin(tobj[2])]) * 0.1
        qbefore = self.ikine(tbefore,qobj)
        qbefore = np.remainder(qbefore, 2*np.pi)
        qbefore[2] = -self.grasp_joint

        print('qbefore is {}'.format(qbefore))
        print('qobj is {}'.format(qobj))
        qpoint.append(qbefore)
        qpoint.append(qobj.copy())
        qobj[2] = self.grasp_joint
        qpoint.append(qobj)

        qend_1 = qobj[0] + np.sign(np.random.randn())*(np.pi/3 + np.random.random()*np.pi/6);
        qend_2 = qobj[1] + np.sign(np.random.randn())*(np.pi/3 + np.random.random()*np.pi/6);
        qend_3 = self.grasp_joint
        qend = np.array([qend_1, qend_2, qend_3])
        tend = self.fkine(qend)
        target_pose = tend[0:2] + np.array([np.cos(tend[2]), np.sin(tend[2])]) * 0
        qpoint.append(qend)

        qTraj = []
        for i in range(len(qpoint)-1):
            qinit = qpoint[i]
            qtarget = qpoint[i+1]

            cq = qinit
            qTraj.append(cq)
            dq = np.array([0,0,0])
            while np.linalg.norm(cq - qtarget) > err_tol:
                dq = kp * (qtarget - cq) - kd * dq
                cq = cq + dq * dtime
                qTraj.append(cq)

        qTraj = np.array(qTraj)
        return qTraj, obj_pose, target_pose

    def runTraj(self, qTraj, isShow=True):
        for i in range(len(qTraj)-1):
            action = qTraj[i+1] - qTraj[i]
            self.step(action)
            if isShow is True:
                self.render()


    # Reinforcement Learning Related Functions
    def step(self, action):
        done = False

        state = self.current_state
        for i in range(3):
            state[0][i] = self.current_state[0][i] + action[i]


        self.sim.data.qpos[0] = state[0][0]
        self.sim.data.qpos[1] = state[0][1]
        self.sim.data.qpos[2] = state[0][2]
        self.sim.data.qpos[3] =  - state[0][2]


        obj_pose = np.array(self.sim.data.get_body_xpos('targetObj'))
        target_pose = np.array(self.sim.data.get_geom_xpos('target_pose'))
        cost = np.linalg.norm(target_pose[0:2] - obj_pose[0:2])

        if cost < 1e-2:
            done = True

        reward = -cost

        self.sim.forward()
        self.sim.step()

        state.append(self.sim.data.get_body_xpos('targetObj'))
        self.current_state = state

        # print('xi pos is {}'.format(self.sim.data.body_xpos))
        # print('obj pos is {}'.format(self.sim.data.get_body_xpos('targetObj')))
        return state, reward, done

    def render(self):
        self.viewer.render()


if __name__ == '__main__':
    # pr_env = PlanarRobotEnv(obj_pose='-1 1 1')
    #
    # grasp_joint = 0.3
    # action = [0,0,-grasp_joint,grasp_joint]
    # pr_env.step(action)
    # action = [0,0,0,0]
    #
    # step = 0.001
    # while 1:
    #     action[0] = step
    #     action[1] = step
    #     state, _, _ = pr_env.step(action)
    #     #print('current state is {}'.format(state))
    #
    #     pr_env.render()
    #     if state[0][1] > 3.14*2/3:
    #         step = 0

    pr_env = PlanarRobotEnv(generate_motion=True)
    pr_env.runTraj(pr_env.qTraj)