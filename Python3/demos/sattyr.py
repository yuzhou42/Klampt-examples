import klampt
from klampt import WorldModel, vis
from klampt.math import vectorops,so3,se3
from klampt.model import ik, collide
import math
from klampt.model import trajectory
from klampt.model import coordinates
from klampt.vis import GLRealtimeProgram
import math
import numpy as np
# from threading import Thread, Lock, RLock

# from klampt.sim import *

# world = WorldModel()
# mode_path = "../../data/sattyr.xml"
# world.readFile(mode_path)
# robot = world.robot(0)
# vis.add("world", world)
# print("links: ",robot.numLinks())                # 12
# print("get config: ", robot.getConfig()) #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# robot.setConfig([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
# print("get config: ", robot.getConfig()) 
# vis.run()

class CartPoleEnv():
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    """
    def __init__(self, world):
        self.robot = world.robot(0)
        self.max_acc = 5.0  # action
        self.gravity = 9.81
        self.masscart = self.robot.link(11).getMass().mass + self.robot.link(8).getMass().mass
        self.total_mass = sum(self.robot.link(i).getMass().mass for i in range(self.robot.numLinks()))
        self.masspole = self.total_mass - self.masscart

        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 30
        self.robot = world.robot(0)
        # self._controlLoopLock = RLock()
        self.state = None

    def step(self,u):
        u = max(min(self.max_acc, u), -u)
        print("u: ", u)
        x, theta, x_dot, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        xacc = (
            u
            + self.masspole
            * sintheta
            * (self.length * (theta_dot ** 2) + self.gravity * costheta)
        ) / (self.masscart + self.masspole * (sintheta ** 2))
        thetaacc = (
            -u * costheta
            - self.masspole * self.length * (theta_dot ** 2) * costheta * sintheta
            - self.total_mass * self.gravity * sintheta
        ) / (self.length * self.masscart + self.masspole * (sintheta ** 2))
        print("x_acc: ",xacc)
        print("theta_acc: ",thetaacc)

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        return np.array((x, theta, x_dot, theta_dot))

class InfiniteLQR:
    def __init__(self, A, B, Q, R):
        self.solve_dare(A, B, Q, R)

    def solve_dare(self, A, B, Q, R):
        """Solve Discrete Algebraic Ricatti Equation"""
        err = float("inf")
        tol = 1e-6
        V = np.random.rand(Q.shape[0], Q.shape[1])
        while err > tol:
            old_V = V
            self.L = -(np.linalg.inv(R + (B.T.dot(V)).dot(B))).dot((B.T.dot(V)).dot(A))
            V = (
                Q
                + (self.L.T.dot(R)).dot(self.L)
                + ((A + B.dot(self.L)).T.dot(V)).dot(A + B.dot(self.L))
            )
            err = np.linalg.norm(old_V - V)


class CartPoleController:
    def __init__(self, CartPoleEnv, state_ref):
        # self.step = 0
        self.env = CartPoleEnv
        self.state_ref = state_ref
        a1 = env.masspole * env.gravity / env.masscart
        a2 = env.total_mass * env.gravity / (env.length * env.masscart)
        self.A = np.eye(4) + env.tau * np.array(
            [[0, 0, 1, 0], [0, 0, 0, 1], [0, a1, 0, 0], [0, a2, 0, 0]]
        )
        self.B = env.tau * np.array(
            [[0], [0], [1 / env.masscart], [1 / env.length * env.masscart]]
        )
        self.Q = np.eye(4)
        self.R = 1
        self.infinite_lqr = InfiniteLQR(self.A, self.B, self.Q, self.R)

    def get_lqr_gain(self):
        return self.infinite_lqr.L

    def act(self, state):
        u = self.get_lqr_gain().dot(state - self.state_ref)
        # self.step += 1
        return u[0]

class GLTest(GLRealtimeProgram):
    def __init__(self,world,sim, cartpoleController, env):
        GLRealtimeProgram.__init__(self,"GLTest")
        self.world = world
        self.sim = sim
        self.robot = world.robot(0)
        # self.pre_x, self.pre_theta, self.pre_x_dot, self.pre_theta_dot = 0,0,0,0
        self.state = None
        self.controller = sim.controller(0)

    def display(self):
        self.sim.updateWorld()
        self.world.drawGL()

    def idle(self):
        # robot.numDrivers(): 6
        # self.dt: 0.02
        print("pre-state: ", self.state)
        self.q = self.controller.getSensedConfig() #self.robot.getConfig()
        self.dq = self.controller.getSensedVelocity()
        x = self.q[0]
        x_dot = self.dq[0]
        theta = self.q[4]
        theta_dot = self.dq[4]
        self.state = (x, theta, x_dot, theta_dot)
        env.state = self.state
        print("state: ", self.state)
        # print("simtime: ",sim.getTime())
        # q=sim.controller(0).getCommandedConfig()
        action = cartpoleController.act(self.state)
        cartpole_config = env.step(action)
        print("cartpole_config: ",cartpole_config)
        wheelCondfig = action
        # TODO: transfer state to klampt configuration
        # q = [0.0, 0.0, cartpole_config[0],  0.0, cartpole_config[1],  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # dq = [ 0.0, 0.0, cartpole_config[2], 0.0, cartpole_config[3],  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        q = [cartpole_config[0], 0.0, 0.0,  0.0, cartpole_config[1],  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        dq = [cartpole_config[2], 0.0, 0.0, 0.0, cartpole_config[3],  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, wheelCondfig, 0.0, 0.0, wheelCondfig]
        # dq = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # sim.controller(0).setPIDCommand(q,dq)
        self.controller.addCubic(q, dq, self.dt)
        # self.controller.setPIDCommand(q, dq)
        # print("getCommandedConfig(): ", self.controller.getCommandedConfig())
        # print("getCommandedVelocity(): ", self.controller.getCommandedVelocity())
        print("getSensedConfig(): ", self.controller.getSensedConfig())
        print("config state: ", self.robot.getConfig())
        print("getSensedVelocity(): ", self.controller.getSensedVelocity())
        print("getCommandedVelocity(): ", self.controller.getCommandedVelocity())
        sim.simulate(self.dt)
        return

if __name__ == "__main__":
    world = klampt.WorldModel()
    res = world.readFile("../../data/sattyr.xml")
    if not res:
        raise RuntimeError("Unable to load world")
    sim = klampt.Simulator(world)
    state_ref = np.array([0, 0, 0, 0])
    env = CartPoleEnv(world)
    env.state = state_ref
    cartpoleController = CartPoleController(env, state_ref)
    GLTest(world,sim,cartpoleController,env).run()
