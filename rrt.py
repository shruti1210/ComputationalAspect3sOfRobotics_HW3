import sim
import pybullet as p
import numpy as np

MAX_ITERS = 10000
delta_q = 0.5

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    """
    Draw a line between two positions corresponding to the input configurations
    :param q_1: configuration 1
    :param q_2: configuration 2
    :param env: environment
    :param color: color of the line, please leave unchanged.
    """
    # obtain position of first point
    #env.set_joint_positions(q_1)
    env.move_joints(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    #env.set_joint_positions(q_2)
    env.move_joints(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)

def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: list of configurations (joint angles) if found a path within MAX_ITERS, else None
    """
    # ========= TODO: Problem 3 ========
    # Implement RRT code here. This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    # Use visualize_path() to visualize the edges in the exploration tree for part (b)
    
    V = np.array([q_init])
    E = []

    for i in range(1, MAX_ITERS):
        q_rand = SemiRandomSample(steer_goal_p, q_goal)
        q_nearest = Nearest(V, E, q_rand)
        q_new = Steer(q_nearest, q_rand, delta_q)
        obstacle_free = ObstacleFree(q_nearest, q_new, env)
        if (obstacle_free==False):
            #print(obstacle_free)
            V = np.vstack((V, q_new))
            E.append((q_nearest, q_new))
            if np.linalg.norm(q_new - q_goal, ord=1) < delta_q:
                V = np.vstack((V, q_goal))
                E.append((q_nearest, q_new))
                print("trying to calculate path")
                path = calculate_path(V, E, q_init, q_goal, env)
                return path
            visualize_path(q_nearest, q_new, env)
    # ==================================
    return None

def SemiRandomSample(steer_goal_p, q_goal):
    if(np.random.random() < steer_goal_p):
        return q_goal
    else:
        #randomly sample from the configuration space, return np.random.choice
        #print(np.random.uniform(-np.pi, np.pi, 6))
        return np.random.uniform(-np.pi, np.pi, 6)

def Nearest(V, E, q_rand):
    min_distance = np.inf
    for q in V:
        distance = np.linalg.norm(q - q_rand, 1)
        if(distance<min_distance):
            min_distance = distance
            q_nearest = q
    return q_nearest

def Steer(q_nearest, q_rand, delta_q):
        delta_q_direction = q_rand - q_nearest
        distance = np.linalg.norm(delta_q_direction)
        if(distance<=delta_q):
            q_new = q_rand
        else:
            q_new = q_nearest + (q_rand - q_nearest)*delta_q_direction/distance
        return q_new

def ObstacleFree(q_nearest, q_new, env):
    return env.check_collision(q_new)

import heapq

def calculate_path(V, E, q_init, q_goal, env):
    def heuristic(q):
        return np.linalg.norm(q_goal - q)
    path = None
    queue = [(heuristic(q_init), q_init)]
    cost_so_far = {tuple(q_init): 0}
    parent = {tuple(q_init): None}

    # optimal path using A* search
    while queue:
        _, q_current = heapq.heappop(queue)

        if np.array_equal(q_current, q_goal):
            path = []
            while q_current is not None:
                path.append(q_current)
                q_current = parent[tuple(q_current)]
            path.reverse()
            break

        for q_parent, q_child in E:
            if np.allclose(q_parent, q_current, atol=1e-8):
                new_cost = cost_so_far[tuple(q_parent)] + np.linalg.norm(q_child - q_parent)
                if tuple(q_child) not in cost_so_far or new_cost < cost_so_far[tuple(q_child)]:
                    cost_so_far[tuple(q_child)] = new_cost
                    priority = new_cost + heuristic(q_child)
                    heapq.heappush(queue, (priority, q_child))
                    parent[tuple(q_child)] = tuple(q_parent)
    #print("found a path")
    if not path:
        return None
    else:
        print("found a path")
        return path


def execute_path(path_conf, env):
    # ========= TODO: Problem 3 ========
    # 1. Execute the path while visualizing the location of joint 5 
    #    (see Figure 2 in homework manual)
    #    You can get the position of joint 5 with:
    #         p.getLinkState(env.robot_body_id, 9)[0]
    #    To visualize the position, you should use sim.SphereMarker
    #    (Hint: declare a list to store the markers)
    # 2. Drop the object (Hint: open gripper, step the simulation, close gripper)
    # 3. Return the robot to original location by retracing the path 
    # Create a list to store the markers
    markers = []

    #print("inside execute path function")

    for q in path_conf:
        env.move_joints(q)
        joint5_position = p.getLinkState(env.robot_body_id, 9)[0]
        # Visualize the position of joint 5
        #print("visualizing joint 5 position")
        marker_position = joint5_position
        marker = sim.SphereMarker(position=marker_position)
        markers.append(marker)

    # Drop the object
    env.open_gripper()
    env.close_gripper()

    path_conf.reverse()
    for q in reversed(path_conf):
        # Set the joint positions of the robot
        env.move_joints(q)

    # ==================================
    return None