import sys
import numpy as np
import casadi
import forcespro
import forcespro.nlp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os.path

nx = 5      # (x, y, theta, v, w)
nu = 2
N_human = 3
N_table = 4
ny = 2
T = 10      # length of planning horizon
# nvar = (nx + nu) * N_human + nu * N_human
nvar = (nx + nu) * N_human
dt = 0.1




def dynamics(z):
    # description of differential drive model with acceleration limit
    # z_t = (u_t^{(1)},..., u_t^{(N)},    x_t^{(N)}, ...,x_t^{(N)},    \Delta u_t^{(1)},...,\Delta u_t^{(N)})
    # TODO : do we need casadi here?
    f = []
    u = casadi.SX.sym('u', nu, N_human)
    x = casadi.SX.sym('x', nx, N_human)
    # deltau = casadi.SX.sym('deltau', nu, N_human)
    for i in range(N_human):
        u[:, i] = z[i * nu: (i + 1) * nu]  # velocity
        x[:, i] = z[N_human * nu + i * nx: N_human * nu + (i + 1) * nx]  # position
        # deltau[:, i] = z[N_human * nu + N_human * nx + i * nu:
        #                  N_human * nu + N_human * nx + (i + 1) * nu]  # acceleration
        # for j in range(nu):
        #    f.append(deltau[j, i] + u[j, i])

        # for j in range(nx):
        #     f.append(x[j, i] + u[j, i])

        f.append(x[0, i] + dt * x[3, i] * casadi.cos(x[2, i]))      # x
        f.append(x[1, i] + dt * x[3, i] * casadi.sin(x[2, i]))      # y
        f.append(x[2, i] + dt * x[4, i])                            # theta
        f.append(x[3, i] + dt * u[0, i])                            # v
        f.append(x[4, i] + dt * u[1, i])                            # w

    return f  # ddelta/dt = phi


def objective(z, p):
    # p : for goal trajectories
    u = casadi.SX.sym('u', nu, N_human)
    x = casadi.SX.sym('x', nx, N_human)
    # deltau = casadi.SX.sym('deltau', nu, N_human)
    goal = casadi.SX.sym('goal', nx, N_human)
    obj = 0.
    # Q = casadi.diag(p[ny * N_human + N_table * ny + N_human + N_table + 1:
    #                   ny * N_human + N_table * ny + N_human + N_table + 1 + ny])
    # R = casadi.diag(p[ny * N_human + N_table * ny + N_human + N_table + 1 + ny:
    #                   ny * N_human + N_table * ny + N_human + N_table + 1 + ny + nu])
    Q = casadi.diag([1., 1., 0.01, 0.1, 0.1])   # state cost weight matrix
    R = casadi.diag([0.1, 0.1])                 # control cost weight matrix

    for i in range(N_human):
        u[:, i] = z[i * nu: (i + 1) * nu]
        x[:, i] = z[N_human * nu + i * nx: N_human * nu + (i + 1) * nx]
        # deltau[:, i] = z[N_human * nu + N_human * nx + i * nu: N_human * nu + N_human * nx + (i + 1) * nu]
        goal[:, i] = p[i * nx: (i + 1) * nx]
        # quadratic objective function
        # obj += casadi.dot((Q @ (x[:, i] - goal[:, i])), x[:, i] - goal[:, i]) + casadi.dot((R @ deltau[:, i]), deltau[:, i])
        obj += casadi.dot((Q @ (x[:, i] - goal[:, i])), x[:, i] - goal[:, i]) + casadi.dot((R @ u[:, i]), u[:, i])
    return obj / N_human


def objectiveN(z, p):
    x = casadi.SX.sym('x', nx, N_human)
    goal = casadi.SX.sym('goal', nx, N_human)
    obj = 0.
    # Qf = casadi.diag(p[ny * N_human + N_table * ny + N_human + N_table + ny + nu + 1:
    #                    ny * N_human + N_table * ny + N_human + N_table + 1 + ny + nu + ny])
    Qf = casadi.diag([20., 20., 0.01, 2., 2.])
    for i in range(N_human):
        x[:, i] = z[N_human * nu + i * nx: N_human * nu + (i + 1) * nx]
        goal[:, i] = p[i * nx: (i + 1) * nx]
        obj += casadi.dot((Qf @ (x[:, i] - goal[:, i])), x[:, i] - goal[:, i])
    return obj / N_human


def inequality(z, p):
    # params : (goal, static_obs_pos)
    f = []
    x = casadi.SX.sym('x', nx, N_human)
    xo = casadi.SX.sym('xo', ny, N_table)
    # w = casadi.SX.sym('w', N_human)
    # wo = casadi.SX.sym('wo', N_table)
    for i in range(N_human):
        x[:, i] = z[N_human * nu + i * nx: N_human * nu + (i + 1) * nx]
        # w[i] = p[N_human * ny + N_table * ny + N_table + i]         # dynamic obstacle radius
    for i in range(N_table):
        xo[:, i] = p[N_human * nx + i * ny: N_human * nx + (i + 1) * ny]
        # wo[i] = p[N_human * ny + N_table * ny + i]                  # static obstacle radius
    # safety = p[N_human * ny + N_table * ny + N_table + N_human]     # safety margin for the dynamic obstacles

    # {n chooses 2} constraints for collisions among dynamic obstacles
    for i in range(N_human):
        for j in range(i + 1, N_human):
            f.append(casadi.dot(x[:ny, i] - x[:ny, j], x[:ny, i] - x[:ny, j]) - (0.18 * 2 + 1.44 ** 2))

    for i in range(N_human):
        for j in range(N_table):
            f.append(casadi.dot(x[:ny, i] - xo[:, j], x[:ny, i] - xo[:, j]) - (0.18 + 0.3) ** 2)
    return f


def generate_pathplanner():
    # Model Definition
    # ----------------

    # Problem dimensions
    model = forcespro.nlp.SymbolicModel(T)
    model.nvar = nvar
    model.neq = nx * N_human
    model.npar = nx * N_human + N_table * ny
    model.nh[0] = 0
    model.nh[1:] = N_table * N_human + N_human * (N_human - 1) // 2

    model.objective = objective
    model.objectiveN = objectiveN
    model.eq = dynamics
    # Indices on LHS of dynamical constraint - for efficiency reasons, make
    # sure the matrix E has structure [0 I] where I is the identity matrix.
    model.E = np.concatenate([np.zeros((nx * N_human, nu * N_human)), np.eye(nx * N_human)],
                             axis=1)
    model.ubidx = range(model.nvar)
    model.lbidx = range(model.nvar)
    for k in range(1, model.N):
        model.ineq[k] = inequality
        model.hl[k] = np.zeros((N_table * N_human + N_human * (N_human - 1) // 2, 1))

    # Initial condition on vehicle states x
    model.xinitidx = range(N_human * nu,
                           N_human * nu + N_human * nx)     # use this to specify on which variables initial conditions

    # Solver generation
    # -----------------

    # Set solver options
    codeoptions = forcespro.CodeOptions('Human_Motion_Gen')
    # codeoptions.nlp.ad_tool = 'casadi-3.5.1'
    codeoptions.maxit = 2000
    codeoptions.printlevel = 0
    codeoptions.optlevel = 3
    codeoptions.cleanup = False
    codeoptions.timing = 1
    codeoptions.overwrite = 1
    codeoptions.nlp.stack_parambounds = True
    codeoptions.parallel = 1
    solver = model.generate_solver(options=codeoptions)

    return solver


def plan(xtable, x0, goal, ep_len):
    #  Code Generation
    # x0's shape = (# dynamic obs, dim)
    # xtable's shape = (# static obs, dim)
    # goal's shape = (# dynamic obs, dim)

    # [max_vel_x, max_vel_theta] : limits of each dynamic obstacle
    max_vel = [0.16, 0.5]
    min_vel = [0., -0.5]

    acc_ub = [1.6, 3.2]    # [acc_lim_x, acc_lim_theta]
    acc_lb = [-1.6, -3.2]
    # velocity bound of the obstacle group
    u_ub = acc_ub * N_human
    u_lb = acc_lb * N_human

    if os.path.exists("Human_Motion_Gen"):
        # print("Solver Exists!")
        solver = forcespro.nlp.Solver.from_directory("./Human_Motion_Gen")
    else:
        solver = generate_pathplanner()

    bounds = np.array([[-2.5, -3.5], [2.0, 3.5]])  # [x_min, y_min], [x_max, y_max]
    # state limits
    xmin = [bounds[0, 0] + 0.18, bounds[0, 1] + 0.18, -np.pi] + min_vel
    xmax = [bounds[1, 0] - 0.18, bounds[1, 1] - 0.18,  np.pi] + max_vel
    # Q, Qf, R = np.array([1, 1]), np.array([1, 1]), np.array([0.01, 0.01])  # cost weights

    Xout = np.zeros((N_human, ep_len + 1, nx))  # (# dynamic obs, episode length + 1, dim)
    Uout = np.zeros((N_human, ep_len, nu))
    Xout1 = np.zeros((N_human, T, nx))
    Uout1 = np.zeros((N_human, T, nu))

    # assert exitflag == 1, "bad exitflag"

    # createPlot(x0, xtable, goal, human_rad, table_rad)

    # set an initial state for MPC at t = 0:
    Xout[:, 0, :] = x0

    # generate trajectories!!!
    for t in range(ep_len):
        xinit1 = []
        for j in range(N_human):
            xinit1.append(Xout[j, t, :])    # set an initial state for MPC at t:
        xinit = np.reshape(xinit1, (1, N_human * nx))

        problem = {"x0": np.zeros((T, nvar)),
                   "xinit": xinit
                   }

        stage_lb = u_lb + N_human * xmin
        stage_ub = u_ub + N_human * xmax

        problem["lb"] = np.concatenate(
            (u_lb, np.tile(stage_lb, T - 1)))
        problem["ub"] = np.concatenate(
            (u_ub, np.tile(stage_ub, T - 1)))

        problem["all_parameters"] = np.tile(np.concatenate((np.reshape(goal, (N_human * nx,)),
                                                            np.reshape(xtable, (N_table * ny,)),
                                                            )), (T,))
        output, exitflag, info = solver.solve(problem)
        # print("took {} iterations and {} seconds to solve the problem {}."\
        # .format(info.it, info.solvetime, exitflag))

        for k in range(T):
            temp = output['x{0:02d}'.format(k + 1)]
            for j in range(N_human):
                Xout1[j, k, :] = temp[N_human * nu + j * nx: N_human * nu + (j + 1) * nx]
                Uout1[j, k, :] = temp[j * nu: (j + 1) * nu]


        Xout[:, t + 1, :] = Xout1[:, 1, :]
        Uout[:, t, :] = Uout1[:, 0, :]

        # updatePlots(Xout, human_rad, i)
        # print("updated")
        # if t == ep_len-1:
        #     print("showed")
        #     plt.show()
        # else:
        #     plt.draw()

    return Xout, Uout
