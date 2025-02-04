
import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt

# 定数
SAMPLING_TIME = 1.0  # サンプリング周期 [s]
SAMPLING_DATA_NUM = 20  # サンプリング点数
m = 1  # 質量 [kg]

# 初期・終端状態
z0 = [0, 0, 0, 0]  # (x, y, x_dot, y_dot)
zf = [20, 30, 0, 0]  # 目標地点

def define_ode():
    """運動方程式（ODE）の定義"""
    z = ca.MX.sym('z', 4)  # 状態変数 [x, y, x_dot, y_dot]
    u = ca.MX.sym('u', 2)  # 制御入力 [Fx, Fy]

    # 運動方程式 dx/dt = v, dv/dt = F/m
    zdot = ca.vertcat(z[2], z[3], u[0] / m, u[1] / m)

    # 評価関数（最適化目的関数）
    L = (z[0] - zf[0])**2 + (z[1] - zf[1])**2 + u[0]**2 + u[1]**2

    dae = {'x': z, 'p': u, 'ode': zdot, 'quad': L}
    opts = {'tf': SAMPLING_TIME}

    return ca.integrator('F', 'cvodes', dae, opts)

def define_optimization_problem(F):
    """非線形最適化問題の定義"""
    w = []  # 最適化変数
    lbw = []  # 最適化変数の下限
    ubw = []  # 最適化変数の上限
    G = []  # 制約
    J = 0  # 評価関数（目的関数）

    # 初期状態
    Xk = ca.MX.sym('X', 4)
    w += [Xk]
    lbw += z0
    ubw += z0

    for i in range(SAMPLING_DATA_NUM):
        # 制御入力の変数を作成
        Uk = ca.MX.sym(f'U{i}', 2)
        w += [Uk]
        lbw += [-1, -1]
        ubw += [1, 1]

        # 1ステップのシミュレーション
        Fk = F(x0=Xk, p=Uk)
        J += Fk["qf"]

        # 次の状態を変数として追加
        Xk = ca.MX.sym(f'X{i+1}', 4)
        w += [Xk]
        G += [Fk['xf'] - Xk]

        if i != SAMPLING_DATA_NUM - 1:
            lbw += [-100, -100, -100, -100]
            ubw += [100, 100, 100, 100]
        else:
            # 最終状態の制約
            lbw += zf
            ubw += zf

    return w, lbw, ubw, G, J

def solve_optimization(w, lbw, ubw, G, J):
    """最適化ソルバーを実行"""
    prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*G)}
    solver = ca.nlpsol('solver', 'ipopt', prob, {"ipopt.print_level": 4})

    sol = solver(x0=0, lbx=lbw, ubx=ubw, lbg=0, ubg=0)
    return sol

def plot_result(sol):
    """最適化結果をプロット"""
    x_star = np.array(sol["x"].elements() + [np.nan] * 2)
    x_star = x_star.reshape(-1, 6)
    x_star = pd.DataFrame(x_star, columns=["x", "y", "x_dot", "y_dot", "Fx", "Fy"])

    _, ax = plt.subplots()
    ax.plot(x_star["x"], x_star["y"])
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    return x_star

# 実行
F = define_ode()  # ODE 定義
w, lbw, ubw, G, J = define_optimization_problem(F)  # 最適化問題の定義
sol = solve_optimization(w, lbw, ubw, G, J)  # 最適化の実行
x_star = plot_result(sol)  # 結果をプロット

print(x_star)
