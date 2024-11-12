import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 定数の設定
m = 1.0      # 質量 (kg)
g = 9.81     # 重力加速度 (m/s^2)
L = 1.0      # 初期y座標 (m)
T = 1.0      # 総時間 (s)
N = 100      # 分割数（時刻のサンプル数）

# 時間刻み幅の計算
dt = T / (N - 1)

# 初期推定の軌道
x_init = np.linspace(0, L, N)
y_init = np.linspace(L, 0, N)
trajectory_init = np.hstack([x_init, y_init])

# ハミルトニアンの定義
def hamiltonian(trajectory):
    x = trajectory[:N]
    y = trajectory[N:]
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    
    # 運動エネルギー T とポテンシャルエネルギー V
    T_kin = 0.5 * m * (vx**2 + vy**2)
    V_pot = m * g * y[:-1]
    
    # ハミルトニアンの総和（時間全体でのエネルギー積分）
    H = np.sum(T_kin + V_pot) * dt
    return H

# 境界条件の設定
def boundary_conditions(trajectory):
    x = trajectory[:N]
    y = trajectory[N:]
    conditions = [
        x[0] - 0,   # 初期位置 x=0
        y[0] - L,   # 初期位置 y=L
        x[-1] - L,  # 終了位置 x=L
        y[-1] - 0   # 終了位置 y=0
    ]
    return conditions

# 制約条件を満たすように制約設定
constraints = {'type': 'eq', 'fun': boundary_conditions}

# 変化過程を保存するためのコールバック関数
trajectory_steps = []  # 各ステップの軌道を保存
def callback(xk):
    trajectory_steps.append(xk.copy())  # 現在の軌道を保存

# 最適化の実行
result = minimize(hamiltonian, trajectory_init, constraints=constraints, method='SLSQP', callback=callback)

# 結果の取得
optimal_trajectory = result.x
x_opt = optimal_trajectory[:N]
y_opt = optimal_trajectory[N:]

# 軌道の図示
plt.figure(figsize=(8, 6))

# 途中経過の軌道を薄い色で表示
alpha_step = 0.011
for i, step in enumerate(trajectory_steps):
    x_step = step[:N]
    y_step = step[N:]
    plt.plot(x_step, y_step, color="gray", alpha=0.3 + alpha_step * i,
             label="途中経過" if i == 0 else "")

# 最終的な最適軌道を濃い色で表示
plt.plot(x_opt, y_opt, color="blue", label='最適軌道', linewidth=2)

# 始点と終点を表示
plt.scatter([0, L], [L, 0], color='red', label='境界条件')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('ハミルトニアン最適化による軌道の途中経過と最適軌道')
plt.legend()
plt.grid()
plt.show()
