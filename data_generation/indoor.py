import numpy as np
from typing import List
from components import Body, Table, Human, Wall
from os import path
import os
import itertools
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as img
import itertools
import argparse
from build_solver import plan


class Indoor:
    def __init__(self,
                 num_tasks,
                 num_tables,
                 num_people,
                 table_radii,
                 human_radii,
                 robot_rad,
                 ep_len=250,
                 dt=0.1
                 ):

        # number of tasks to generate
        # to reduce the computational burden, we generate multiple scenarios in advance, and sample tasks from these
        self.dt = dt
        self.num_tasks = num_tasks
        self.ep_len = ep_len
        self.robot_rad = robot_rad

        # restaurant map configuration
        # here we assume that configuration space is simply a 2-dim cube [-10, 10]^2
        # self.upper_bound = np.array([10., 10.])
        # self.lower_bound = np.array([-10., -10.])

        # ---------------------------------- Walls ---------------------------------
        self.lower_bound = np.array([-2.5 + self.robot_rad, -3.5 + self.robot_rad])
        self.upper_bound = np.array([ 2.0 - self.robot_rad,  3.5 - self.robot_rad])
        # ------------------------------ Tables Setup ------------------------------
        self.num_tables = num_tables
        self.table_radii = table_radii


        # shape = (# of tasks, # of people/tables, episode length + 1, 2)
        human_trajectories, table_positions = self.load_data()

        assert table_positions.shape[0] == num_tasks
        assert human_trajectories.shape[0] == num_tasks

        # p = np.random.permutation(num_tasks)

        # table_positions = table_positions[p]
        # human_trajectories = human_trajectories[p]

        self.tables_in_task = [
            [
                Table(center=table_positions[task_idx, table_idx, :], radius=table_radii)
             for table_idx in range(num_tables)]    # tables for each task
            for task_idx in range(num_tasks)
        ]
        self.tables = []
        # --------------------------------------------------------------------------

        # ------------------------------ People Setup ------------------------------

        self.num_people = num_people
        self.human_radii = human_radii

        self.people_in_task = [
            [Human(trajectory=human_trajectories[task_idx, human_idx, :, :], radius=self.human_radii)
             for human_idx in range(num_people)]
            for task_idx in range(num_tasks)
        ]

        self.people = []  # placeholder for human objects
        # --------------------------------------------------------------------------
        return

    def load_data(self):
        if not (os.path.exists(path.join(path.dirname(__file__), "data/dynamic_obs_states.npy"))
                and os.path.exists(path.join(path.dirname(__file__), "data/static_obs_states.npy"))):
            os.makedirs(path.join(path.dirname(__file__), "data"), exist_ok=True)
            """
            xtable = np.array([
                [-1.7, -1.5],
                [1.7, -1.5],
                [-1.7, 2.5],
                [1.7, 2.5]
            ])
            """
            human_trajectories = []
            human_controls = []
            table_positions = []
            np.random.seed(20220804)

            for task in range(self.num_tasks):
                ######
                #4##1#
                ######
                #3##2#
                ######

                xtable = []

                table_pos_ub = [np.array([1.5, 2.5]), np.array([1.5, -1.0]), np.array([-1., -1.]), np.array([-1., 2.5])]
                table_pos_lb = [np.array([1., 1.]), np.array([1., -2.5]), np.array([-1.5, -2.2]), np.array([-1.5, 1.])]

                
                xtable = np.array([(ub - lb) * np.random.rand(2) + lb for lb, ub in zip(table_pos_lb, table_pos_ub)])

                #####
                ##1##
                #3#2#
                #G###
                #####

                start_pos_lb = [np.array([-0.5, 1.7]), np.array([1.0, -0.5]), np.array([-2.0, -0.5])]
                start_pos_ub = [np.array([0.5, 2.7]), np.array([1.5, 0.5]), np.array([-1.5, 0.5])]


                # goal_pos_lb = [np.array([-0.5, -3.0]), np.array([-2.0, -0.5]), np.array([1.0, -0.5])]
                # goal_pos_ub = [np.array([0.5, -2.4]), np.array([-1.5, 0.5]), np.array([1.5, 0.5])]
                goal_pos_lb = [np.array([1.0, -0.5]), np.array([-0.5, -3.0]), np.array([1.0, -0.5])]
                goal_pos_ub = [np.array([1.5, 0.5]), np.array([0.5, -2.4]), np.array([1.5, 0.5])]

                start1, start2, start3 = [lb + (ub - lb) * np.random.rand(2) for lb, ub in zip(start_pos_lb, start_pos_ub)]
                goal1, goal2, goal3 = [lb + (ub - lb) * np.random.rand(2) for lb, ub in zip(goal_pos_lb, goal_pos_ub)]

                x0 = np.array([
                    np.concatenate([start1, np.array([-np.pi / 2., 0., 0., ])]),
                    # np.concatenate([start2, np.array([np.pi / 2., 0., 0., ])]),
                    np.concatenate([start2, np.array([-0.75 * np.pi, 0., 0., ])]),
                    np.concatenate([start3, np.array([0., 0., 0., ])]),
                    ])


                goal = np.array([
                    np.concatenate([goal1, np.array([-np.pi / 2., 0., 0., ])]),
                    np.concatenate([goal2, np.array([np.pi / 2., 0., 0., ])]),
                    np.concatenate([goal3, np.array([0., 0., 0., ])]),
                    ])
                """
                x_start = 2.4 * np.random.rand() - 1.2
                x_goal = 2.4 * np.random.rand() - 1.2

                y_start = 3.0 * np.random.rand() - 1.0
                y_goal = 3.0 * np.random.rand() - 1.0
                

                x0 = np.array([
                    [-2.2, y_start, -np.pi / 2., 0., 0.],
                    [x_start, 3.2, 0., 0., 0.]
                ])
                goal = np.array([
                    [1.7, y_goal, -np.pi / 2., 0., 0.],
                    [x_goal, -2.7, 0., 0., 0.]
                ])
                """
                # shape = (# human, episode len, dim)
                Xout, Uout = plan(xtable=xtable, x0=x0, goal=goal,  ep_len=self.ep_len)
                print('generating task {}...'.format(task))
                human_trajectories.append(Xout)
                human_controls.append(Uout)
                table_positions.append(xtable)

            # table_positions = np.repeat(xtable[np.newaxis, ...], repeats=self.num_tasks, axis=0)
            np.save(path.join(path.dirname(__file__), "data/dynamic_obs_states.npy"), human_trajectories)
            np.save(path.join(path.dirname(__file__), "data/static_obs_states.npy"), table_positions)
            np.save(path.join(path.dirname(__file__), 'data/dynamic_obs_controls.npy'), human_controls)

        human_trajectories = np.load(path.join(path.dirname(__file__), "data/dynamic_obs_states.npy"))
        table_positions = np.load(path.join(path.dirname(__file__), "data/static_obs_states.npy"))

        return human_trajectories, table_positions

    def reset(self):
        # bring back people to their spawn point
        for human in self.people:
            human.reset()
        # self.human_center_list = [[-0.9, 0.], [0., 0.9]]

    def reset_task(self, idx):
        self.people = self.people_in_task[idx]
        self.tables = self.tables_in_task[idx]

    def sim(self):
        # 1-step simulation of obstacle motions
        for human in self.people:
            human.sim()
        #     self.human_center_list[i] = human.center      # update position of each guest
        return
    """
    def sample_from_free_space(self, margin=0.1):
        # Sample a point from the free space uniformly
        while True:
            rnd = self.total_volume * np.random.rand()
            sum_vol = 0.
            candidate = None
            for i in range(self.num_partitions):
                sum_vol += self.space_volume[i]
                if rnd < sum_vol:
                    candidate = self.free_space_sampling[i][0] + (
                            self.free_space_sampling[i][1] - self.free_space_sampling[i][0]) * np.random.rand(2)
                    break
            collide = False
            for o in itertools.chain(self.tables, self.people):
                collide = (o.distance(candidate) <= self.robot_rad + margin)
                if collide:
                    break
            if not collide:
                return candidate
    """
    @property
    def obstacle_list(self) -> List[Body]:
        # list of every existing obstacles
        return self.tables + self.people

    @property
    def table_list(self) -> List[Table]:
        # list of every existing obstacles
        return self.tables[:]

    @property
    def human_list(self) -> List[Human]:
        # list of every existing obstacles
        return self.people[:]

    """
    @property
    def wall_list(self) -> List[Wall]:
        return self.walls[:]
    """
    @property
    def human_vector(self) -> np.ndarray:
        # return the real-time positions of all people as a flat numpy array
        human_center_list = [human.center for human in self.people]
        return np.reshape(np.array(human_center_list), self.num_people * 2)

    @property
    def table_vector(self) -> np.ndarray:
        # return the center positions of all tables as a flat numpy array
        table_center_list = [table.center for table in self.tables]
        return np.reshape(np.array(table_center_list), self.num_tables * 2)


    def render(self):
        fig = plt.gcf()
        ax = fig.gca()
        """
        background = img.imread(path.join(path.dirname(__file__), "assets/map.png"))
        ax.imshow(background,
                  cmap='gray',
                  extent=[-2.5, 2., -3., 3.5]
                  )
        """
        """
        for i, wall in enumerate(self.walls):
            wall_center = wall.lb + (wall.ub - wall.lb) / 2.
            ax.text(x=wall_center[0], y=wall_center[1], s='{}'.format(i))
        """
        #ax.add_patch(Rectangle((self.room_1[0], self.room_1[1]), self.room_1[2]-self.room_1[0], self.room_1[3]-self.room_1[1], fc='none', ec='k', lw=1))
        #ax.add_patch(Rectangle((self.room_2[0], self.room_2[1]), self.room_2[2]-self.room_2[0], self.room_2[3]-self.room_2[1], fc='none', ec='k', lw=1))

        """
        for i in range(len(self.lb_list)):
            ax.add_patch(Rectangle((self.lb_list[i][0], self.lb_list[i][1]), self.ub_list[i][0]-self.lb_list[i][0], self.ub_list[i][1]-self.lb_list[i][1],
                                   fc='none', ec='b', ls='--', lw=1))
        """
        """
        for o in self.tables:
            obstacle = plt.Circle(o.center,
                                  o.radius,
                                  color='tab:red',
                                  )
            ax.add_artist(obstacle)
        """
        ax.set_xlim(-2.5, 2.0)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        fig.tight_layout()
        # ax.grid()
        return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', required=False, default=10, type=int)
    parser.add_argument('--num_tables', required=False, default=1, type=int)
    parser.add_argument('--num_people', required=False, default=0, type=int)
    parser.add_argument('--table_radii', required=False, default=0.5, type=float)
    parser.add_argument('--robot_rad', required=False, default=0.455, type=float)
    parser.add_argument('--human_radii', required=False, default=0.1, type=float)
    parser.add_argument('--ep_len', required=False, default=200, type=int)
    parser.add_argument('--dt', required=False, default=0.1, type=float)
    parser.add_argument('--print_map', action='store_true')
    args = parser.parse_args()

    map = Indoor(
        num_tables=args.num_tables,
        num_tasks=args.num_tasks,
        num_people=args.num_people,
        table_radii=args.table_radii,
        human_radii=args.human_radii,
        robot_rad=args.robot_rad,
        ep_len=args.ep_len,
        dt=args.dt)

