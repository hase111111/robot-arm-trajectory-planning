"""
This file contains the forward kinematics of a 2-link robot.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

import casadi as ca

LENGTH1 = 1
LENGTH2 = 0.7
LENGTH3 = 0.9
TARGETPOS = (0, 2.6)

# FK
def fk(ang1, ang2, ang3):
    """
    Forward kinematics of a 3-link robot
    """
    x = LENGTH1 * ca.cos(ang1) + LENGTH2 * ca.cos(ang1 + ang2) + LENGTH3 * ca.cos(ang1 + ang2 + ang3)
    y = LENGTH1 * ca.sin(ang1) + LENGTH2 * ca.sin(ang1 + ang2) + LENGTH3 * ca.sin(ang1 + ang2 + ang3)

    return x, y


def val(ang1, ang2, ang3):
    """
    Forward kinematics of a 2-link robot
    """
    x, y = fk(ang1, ang2, ang3)

    return (x - TARGETPOS[0])**2 + (y - TARGETPOS[1])**2

theta1: ca.SX = ca.SX.sym("theta1")
theta2: ca.SX = ca.SX.sym("theta2")
theta3: ca.SX = ca.SX.sym("theta3")

cost = val(theta1, theta2, theta3)

g = ca.vertcat(theta1, theta2, theta3)

# nlpsolver
nlp = {
    "x": ca.vertcat(theta1, theta2, theta3),
    "f": cost,
    "g": g,
}

solver = ca.nlpsol("solver", "ipopt", nlp)

# initial guess, at random
theta_init = [random.uniform(-ca.pi, ca.pi) for _ in range(3)]

upper_bound = [ca.pi * 2, ca.pi * 2, ca.pi * 2]
lower_bound = [-ca.pi * 2, -ca.pi * 2, -ca.pi * 2]

# solve
res = solver(x0=theta_init, lbg=lower_bound, ubg=upper_bound)

# get result
theta_opt = res["x"]
theta_opt_deg = np.rad2deg(theta_opt)

print(f"Optimal angles: {theta_opt}")
print(f"Optimal cost: {res['f']}")

pos = fk(theta_opt[0], theta_opt[1], theta_opt[2])
print("Optimal position:", pos)

# plot
res_origin = [0, 0]
res_target = [TARGETPOS[0], TARGETPOS[1]]   
res_pos1 = [LENGTH1 * np.cos(float(theta_opt[0])), LENGTH1 * np.sin(float(theta_opt[0]))]
res_pos2 = [res_pos1[0] + LENGTH2 * np.cos(float(theta_opt[0] + theta_opt[1])), res_pos1[1] + LENGTH2 * np.sin(float(theta_opt[0] + theta_opt[1]))]
res_pos3 = [res_pos2[0] + LENGTH3 * np.cos(float(theta_opt[0] + theta_opt[1] + theta_opt[2])), res_pos2[1] + LENGTH3 * np.sin(float(theta_opt[0] + theta_opt[1] + theta_opt[2]))]

plt.plot(res_origin[0], res_origin[1], 'bo')
plt.plot([res_origin[0], res_pos1[0], res_pos2[0], res_pos3[0]], [res_origin[1], res_pos1[1], res_pos2[1], res_pos3[1]], 'b-')
plt.plot(res_target[0], res_target[1], 'rx')
plt.plot(res_pos3[0], res_pos3[1], 'ro')
plt.axis('equal')
plt.grid()

# -2.5 ~ 2.5
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)

plt.show()

