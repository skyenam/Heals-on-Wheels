import numpy as np
from geometry import Body, CircularBody, RectangularBody, StaticBody, DynamicBody


class Robot(CircularBody, DynamicBody):
    # Definition of robot
    def __init__(self, radius, max_lin_vel, max_ang_vel):
        super(Robot, self).__init__(center=None, radius=radius)
        self._th = None
        self._control = None

        self._max_lin_vel = max_lin_vel
        self._max_ang_vel = max_ang_vel

    def sim(self, control, dt):
        """
        One-step simulation of a robot which follows
        x_{k+1} = x_k + F(x_k, u_k + epsilon^u_k) Delta t + epsilon^x_k,    k = 0, 1, ...,
        where epsilon^u_k & epsilon^x_k's are i.i.d. normally distributed.
        Here we assume that the vector field F is given as
        F(x, u) = [v cos(th), v sin(th), alpha]
        with x = [x1, x2, th] & u = [v, alpha].
        """
        # if control_noise:
        #     control += np.clip(0.05 * np.random.randn(2), -0.1, 0.1)         # truncated gaussian noise
        lin_vel, angular_vel = control
        self._center += lin_vel * dt * np.array([np.cos(self._th), np.sin(self._th)])
        self._th += dt * angular_vel
        self._th = (self._th + np.pi) % (2. * np.pi) - np.pi        # normalize angle : \theta \in [-\pi, \pi)
        # if state_noise:
        #     self._center += np.clip(0.05 * np.random.randn(2), -0.1, 0.1)
        #     self._th += np.clip(0.05 * np.random.randn(), -0.1, 0.1)
        return

    def reset(self, init_pos, init_th):
        self._center = init_pos.copy()
        self._th = init_th

    def contains(self, pt) -> bool:
        return CircularBody.contains(self, pt)

    def entry_time(self, pt, ray) -> float:
        return CircularBody.entry_time(self, pt, ray)

    def distance(self, pt):
        return CircularBody.distance(self, pt)

    @property
    def state(self):
        return np.array([self._center[0], self._center[1], self._th])


class SecondOrderRobot(CircularBody, DynamicBody):
    # Definition of robot
    def __init__(self, radius, min_lin_vel, min_ang_vel, max_lin_vel, max_ang_vel):
        super(SecondOrderRobot, self).__init__(center=None, radius=radius)

        # state variables : (x, y, theta, v, w)
        self._th = None
        self._control = None

        self._lin_vel = None
        self._ang_vel = None

        # velocity constraints
        # v_min <= v <= v_max
        # w_min <= w <= w_max
        self._min_lin_vel = min_lin_vel
        self._min_ang_vel = min_ang_vel

        self._max_lin_vel = max_lin_vel
        self._max_ang_vel = max_ang_vel

        # acceleration constraints

    def sim(self, control, dt, state_noise=False, control_noise=False):
        """
        One-step simulation of a robot which follows
        x_{k+1} = x_k + F(x_k, u_k + epsilon^u_k) Delta t + epsilon^x_k,    k = 0, 1, ...,
        where epsilon^u_k & epsilon^x_k's are i.i.d. normally distributed.
        Here we assume that the vector field F is given as
        F(x, u) = [v cos(th), v sin(th), w, a, alpha]
        with x = [x1, x2, th, v, w], u = [a, alpha].
        """
        lin_acc, ang_acc = control
        if control_noise:
            # actuator noise N(0, sigma^2)
            lin_acc += np.clip(0.05 * np.random.randn(), -0.1, 0.1)
            ang_acc += np.clip(0.05 * np.random.randn(), -0.1, 0.1)

        # controlled difference equation for the state variables
        # TODO : use Runge-Kutta for ODE simulation. (In this case, the MPC model becomes biased...)

        self._center += self._lin_vel * dt * np.array([np.cos(self._th), np.sin(self._th)])
        self._th += dt * self._ang_vel
        self._th = (self._th + np.pi) % (2. * np.pi) - np.pi        # normalize angle : \theta \in [-\pi, \pi)

        self._lin_vel += dt * lin_acc
        self._ang_vel += dt * ang_acc

        # clip velocity variables w.r.t. constraints
        self._lin_vel = np.clip(self._lin_vel, self._min_lin_vel, self._max_lin_vel)
        self._ang_vel = np.clip(self._ang_vel, self._min_ang_vel, self._max_ang_vel)

        if state_noise:
            # system noise
            self._center += np.clip(0.05 * np.random.randn(2), -0.1, 0.1)
            self._th += np.clip(0.05 * np.random.randn(), -0.1, 0.1)
            self._lin_vel += np.clip(0.05 * np.random.randn(), -0.1, 0.1)
            self._ang_vel += np.clip(0.05 * np.random.randn(), -0.1, 0.1)
        return

    def reset(self, init_pos, init_th):
        # initialize state variables
        self._center = init_pos.copy()
        self._th = init_th
        self._lin_vel = 0.
        self._ang_vel = 0.

    @property
    def state(self):
        return np.array([self._center[0], self._center[1], self._th, self._lin_vel, self._ang_vel])

    def contains(self, pt) -> bool:
        return CircularBody.contains(self, pt)

    def entry_time(self, pt, ray) -> float:
        return CircularBody.entry_time(self, pt, ray)

    def distance(self, pt):
        return CircularBody.distance(self, pt)






# obstacles of the environment
class Human(CircularBody, DynamicBody):
    def __init__(self, radius, trajectory):
        center = trajectory[0, :2]       # center at initial time step
        super(Human, self).__init__(center, radius)
        self._trajectory = trajectory
        self._step = None
        self._state = trajectory[0]

    def sim(self):
        # 1-step simulation of a human
        self._step += 1
        self._center = self._trajectory[self._step, :2]
        self._state = self._trajectory[self._step]

    def reset(self):
        self._step = 0
        self._center = self._trajectory[0, :2]

    @property
    def state(self):
        return np.copy(self._state)

    def contains(self, pt) -> bool:
        return CircularBody.contains(self, pt)

    def entry_time(self, pt, ray) -> float:
        return CircularBody.entry_time(self, pt, ray)

    def distance(self, pt):
        return CircularBody.distance(self, pt)


class Table(CircularBody, StaticBody):
    # table modelled as circular obstacle
    def __init__(self, center, radius):
        super(Table, self).__init__(center, radius)

    def contains(self, pt) -> bool:
        return CircularBody.contains(self, pt)

    def entry_time(self, pt, ray) -> float:
        return CircularBody.entry_time(self, pt, ray)

    def distance(self, pt):
        return CircularBody.distance(self, pt)


class Wall(RectangularBody, StaticBody):
    # table modelled as circular obstacle
    def __init__(self, lb, ub):
        super(Wall, self).__init__(lb, ub)

    def contains(self, pt) -> bool:
        return RectangularBody.contains(self, pt)

    def entry_time(self, pt, ray) -> float:
        return RectangularBody.entry_time(self, pt, ray)

    def distance(self, pt):
        return RectangularBody.distance(self, pt)


