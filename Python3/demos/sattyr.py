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
    def __init__(self, world, sim):
        self.sim = sim
        self.max_acc = 5.0  # action
        self.gravity = 9.81
        self.masscart = 1
        self.masspole = 17
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length

        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 30
        self.robot = world.robot(0)


        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        # high = np.array(
        #     [
        #         self.x_threshold * 2,
        #         np.finfo(np.float32).max,
        #         self.theta_threshold_radians * 2,
        #         np.finfo(np.float32).max,
        #     ],
        #     dtype=np.float32,
        # )

        # self.action_space = spaces.Box(
        #     low=-self.max_acc, high=self.max_acc, shape=(1,), dtype=np.float32
        # )
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # self.seed()
        # self.viewer = None
        self.state = None
        self.pre_x, self.pre_theta, self.pre_x_dot, self.pre_theta_dot = 0,0,0,0

        # self.steps_beyond_done = None

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self,u):
        # u = max(min(self.max_acc, u), -u)
        self.config = self.robot.getConfig()
        x = self.config[0]
        x_dot = (self.config[0] - self.pre_x)/self.tau
        theta = self.config[4]
        theta_dot = (theta - self.pre_x_dot)/self.tau
        self.pre_x, self.pre_theta, self.pre_x_dot, self.pre_theta_dot = x, x_dot, theta, theta_dot
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

        self.state = (x, theta, x_dot, theta_dot)
        return np.array(self.state)

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
    def __init__(self, CartPoleEnv, state, state_ref):
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
    def __init__(self,world,sim, controller, env):
        GLRealtimeProgram.__init__(self,"GLTest")
        self.world = world
        self.sim = sim
        self.robot = world.robot(0)


    def display(self):
        self.sim.updateWorld()
        self.world.drawGL()

    def idle(self):
        # print("simtime: ",sim.getTime())
        # print("dt: ",self.dt)
        # q=sim.controller(0).getCommandedConfig()
        # print("q: ",q)
        action = controller.act(env.state)
        state = env.step(action)
        # TODO: transfer state to klampt configuration
        q = [state[0], 0.0, 0.0, 0.0, state[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sim.controller(0).setMilestone(q)
        # self.robot.setConfig(q) 
        sim.simulate(self.dt)

        print("state: ", self.robot.getConfig())
        return

if __name__ == "__main__":
    world = klampt.WorldModel()
    res = world.readFile("../../data/sattyr.xml")
    if not res:
        raise RuntimeError("Unable to load world")
    sim = klampt.Simulator(world)
    state_ref = np.array([0, 0, 0, 0])
    env = CartPoleEnv(world, sim)
    env.state = state_ref
    controller = CartPoleController(env, env.state, state_ref)
    GLTest(world,sim,controller,env).run()
