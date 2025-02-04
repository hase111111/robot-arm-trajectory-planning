"""
This file contains the forward kinematics of a 2-link robot.
"""

import numpy as np
import matplotlib.pyplot as plt

import casadi as ca


# FK
def fk(ang1, ang2):
    """
    Forward kinematics of a 2-link robot
    """
    x = LENGTH1 * ca.cos(ang1) + LENGTH2 * ca.cos(ang1 + ang2)
    y = LENGTH1 * ca.sin(ang1) + LENGTH2 * ca.sin(ang1 + ang2)

    return x, y


# Parameters
LENGTH1 = 1
LENGTH2 = 0.7
TARGET_POS = (-1.5, 0.0000001)

SIMULATION_TIME = 50.0
DELTA_TIME = 1.0
STEP = int(SIMULATION_TIME / DELTA_TIME)
TIME_SQUENCE = np.linspace(0, SIMULATION_TIME, STEP)

OBSTACLE_POS = [[0.0, -1.5], [1, 2]]
OBSTACLE_RADIUS = [1.0, 0.7]


# Cost function
def dist_cost(ang1s, ang2s):
    """
    Cost function of a 2-link robot
    """
    x, y = fk(ang1s[-1], ang2s[-1])
    return (x - TARGET_POS[0]) ** 2 + (y - TARGET_POS[1]) ** 2


# 加速度の滑らかさを考慮したコスト関数
def smooth_cost(ang1s, ang2s):
    """
    Cost function of a 2-link robot
    """
    diff1 = [0] + [ang1s[i + 1] - ang1s[i] for i in range(STEP - 1)] + [0]
    diff2 = [0] + [ang2s[i + 1] - ang2s[i] for i in range(STEP - 1)] + [0]
    ddiff1 = [0] + [diff1[i + 1] - diff1[i] for i in range(STEP)] + [0]
    ddiff2 = [0] + [diff2[i + 1] - diff2[i] for i in range(STEP)] + [0]
    dddiff1 = [ddiff1[i + 1] - ddiff1[i] for i in range(STEP + 1)]
    dddiff2 = [ddiff2[i + 1] - ddiff2[i] for i in range(STEP + 1)]

    sum_v = 0

    for i in range(STEP + 1):
        sum_v += dddiff1[i] ** 2 + dddiff2[i] ** 2

    return sum_v


# 障害物との距離を考慮した制約
def obstacle_cost(ang1s, ang2s):
    min_dist = ca.inf
    for i in range(STEP):
        for j, v in enumerate(OBSTACLE_POS):
            x, y = fk(ang1s[i], ang2s[i])
            dist = ca.sqrt((x - v[0]) ** 2 + (y - v[1]) ** 2) - OBSTACLE_RADIUS[j]
            min_dist = ca.if_else(dist < min_dist, dist, min_dist)

            x1, y1 = (LENGTH1 * ca.cos(ang1s[i]), LENGTH1 * ca.sin(ang1s[i]))
            dist = ca.sqrt((x1 - v[0]) ** 2 + (y1 - v[1]) ** 2) - OBSTACLE_RADIUS[j]
            min_dist = ca.if_else(dist < min_dist, dist, min_dist)
    return min_dist


# Optimization
theta1 = ca.SX.sym("theta1", STEP)
theta2 = ca.SX.sym("theta2", STEP)

cost = dist_cost(theta1, theta2) + smooth_cost(theta1, theta2)

g = ca.vertcat(theta1, theta2, obstacle_cost(theta1, theta2))

# nlpsolver
nlp = {
    "x": ca.vertcat(theta1, theta2),
    "f": cost,
    "g": g,
}

solver = ca.nlpsol("solver", "ipopt", nlp)

# initial guess, 0, 0
theta_init = [0.0] * STEP * 2

upper_bound = [0] + [ca.pi] * (STEP - 1) + [0] + [ca.pi] * (STEP - 1) + [ca.inf]
lower_bound = (
    [0] + [-ca.pi] * (STEP - 1) + [0] + [-ca.pi] * (STEP - 1) + [0.0]
)

# solve
res = solver(x0=theta_init, lbg=lower_bound, ubg=upper_bound)

# get result
theta_opt = res["x"]
cost_opt = res["f"]

print(f"Optimal cost: {cost_opt}")
print(f"Optimal angles: {theta_opt}")

# plot
fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(STEP):
    angle1 = float(theta_opt[i])
    angle2 = float(theta_opt[i + STEP])
    origin = (0, 0)
    pos1 = (LENGTH1 * np.cos(angle1), LENGTH1 * np.sin(angle1))
    pos2 = (
        pos1[0] + LENGTH2 * np.cos(angle1 + angle2),
        pos1[1] + LENGTH2 * np.sin(angle1 + angle2),
    )

    color = (float(i) / STEP, 0, 1.0 - float(i) / STEP)
    ax.plot(
        [origin[0], pos1[0], pos2[0]], [origin[1], pos1[1], pos2[1]], "-", color=color
    )
    ax.plot(
        [origin[0], pos1[0], pos2[0]], [origin[1], pos1[1], pos2[1]], "o", color=color
    )

ax.plot(TARGET_POS[0], TARGET_POS[1], "bo")
ax.plot(0, 0, "go")

# 円を描画
for i in range(len(OBSTACLE_POS)):
    circle = plt.Circle(
        OBSTACLE_POS[i], OBSTACLE_RADIUS[i], color="r", fill=False, linestyle="--"
    )
    ax.add_artist(circle)
ax.set_aspect("equal", "box")

plt.xlabel("x")
plt.ylabel("y")
plt.title("2-link robot")
plt.grid()

# -3.0 から 3.0 までの範囲を表示
plt.xlim(-3.0, 3.0)
plt.ylim(-3.0, 3.0)

plt.show()

# angle-t plots, dangle-t plots, ddangle-t plots
fig, ax = plt.subplots(3, 1, figsize=(10, 10))

theta_plot = np.rad2deg(theta_opt)

ax[0].plot(TIME_SQUENCE, theta_plot[:STEP], label="theta1")
ax[0].plot(TIME_SQUENCE, theta_plot[STEP:], label="theta2")
ax[0].set_xlabel("time")
ax[0].set_ylabel("angle")
ax[0].legend()
ax[0].grid()

theta_plot = np.array(theta_opt).reshape(2, STEP).T
dtheta_plot1 = np.gradient(theta_plot[:, 0])
dtheta_plot2 = np.gradient(theta_plot[:, 1])

ax[1].plot(TIME_SQUENCE, dtheta_plot1, label="dtheta1")
ax[1].plot(TIME_SQUENCE, dtheta_plot2, label="dtheta2")
ax[1].set_xlabel("time")
ax[1].set_ylabel("dangle")
ax[1].legend()
ax[1].grid()

ddtheta_plot1 = np.gradient(dtheta_plot1)
ddtheta_plot2 = np.gradient(dtheta_plot2)
ax[2].plot(TIME_SQUENCE, ddtheta_plot1, label="ddtheta1")
ax[2].plot(TIME_SQUENCE, ddtheta_plot2, label="ddtheta2")
ax[2].set_xlabel("time")
ax[2].set_ylabel("ddangle")
ax[2].legend()
ax[2].grid()

plt.show()
