
import numpy as np
import matplotlib.pyplot as plt

# 2リンクマニピュレータのパラメータ
L1 = 1.0  # リンク1の長さ
L2 = 1.0  # リンク2の長さ

# 初期姿勢と目標姿勢
q_init = np.array([0.0, 0.0])  # 初期関節角度 [theta1, theta2]
q_goal = np.array([np.pi / 2, np.pi / 2])  # 目標関節角度

# シミュレーションのパラメータ
num_points = 100  # 軌道の離散化点数
alpha = 0.01  # 学習率（勾配の更新量）
num_iterations = 1000  # 最適化の反復回数

# 障害物の定義 (円形障害物)
obstacle_center = np.array([1.0, 1.0])
obstacle_radius = 0.5

# 初期軌道（直線補間）
q_traj = np.linspace(q_init, q_goal, num_points)


# 順運動学
def forward_kinematics(q):
    """2リンクマニピュレータの順運動学"""
    x = L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1])
    y = L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])
    return np.array([x, y])


# ヤコビアンの計算
def jacobian(q):
    """2リンクマニピュレータのヤコビアン"""
    j11 = -L1 * np.sin(q[0]) - L2 * np.sin(q[0] + q[1])
    j12 = -L2 * np.sin(q[0] + q[1])
    j21 = L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1])
    j22 = L2 * np.cos(q[0] + q[1])
    return np.array([[j11, j12], [j21, j22]])


# コスト関数の定義
def smoothness_cost(q_traj):
    """軌道のスムーズさを評価するコスト関数"""
    cost = 0
    for i in range(1, len(q_traj) - 1):
        cost += np.sum((q_traj[i + 1] - 2 * q_traj[i] + q_traj[i - 1]) ** 2)
    return cost


def obstacle_cost(q_traj):
    """軌道の障害物回避コスト"""
    cost = 0
    for q in q_traj:
        ee_pos = forward_kinematics(q)  # エンドエフェクタの位置
        dist = np.linalg.norm(ee_pos - obstacle_center)
        if dist < obstacle_radius:
            cost += (obstacle_radius - dist) ** 2  # 障害物の内側ではペナルティ
    return cost


def total_cost(q_traj):
    """総コスト（スムーズさ + 障害物回避）"""
    return smoothness_cost(q_traj)


# コスト関数の勾配計算
def compute_gradient(q_traj):
    """軌道に対するコストの勾配を計算"""
    grad = np.zeros_like(q_traj)

    # スムーズさに関する勾配
    for i in range(1, len(q_traj) - 1):
        grad[i] += 2 * (q_traj[i] - 2 * q_traj[i - 1] + q_traj[i - 2])

    # 障害物に関する勾配
    for i, q in enumerate(q_traj):
        ee_pos = forward_kinematics(q)
        dist = np.linalg.norm(ee_pos - obstacle_center)
        if dist < obstacle_radius:
            jacob = jacobian(q)  # ヤコビアンを用いた勾配変換
            ee_grad = 2 * (obstacle_radius - dist) * (ee_pos - obstacle_center) / dist
            grad[i] += np.dot(jacob.T, ee_grad)  # 関節空間での勾配に変換

    return grad


# 最適化ループ
for iteration in range(num_iterations):
    grad = compute_gradient(q_traj)
    q_traj -= alpha * grad  # 勾配による更新

    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Cost: {total_cost(q_traj)}")


# 軌道のプロット
def plot_robot(q, ax):
    """マニピュレータの描画"""
    x0, y0 = 0, 0
    x1 = L1 * np.cos(q[0])
    y1 = L1 * np.sin(q[0])
    x2 = x1 + L2 * np.cos(q[0] + q[1])
    y2 = y1 + L2 * np.sin(q[0] + q[1])

    ax.plot([x0, x1, x2], [y0, y1, y2], "-o", color="blue")
    ax.plot(obstacle_center[0], obstacle_center[1], "ro", markersize=10)
    circle = plt.Circle(obstacle_center, obstacle_radius, color="r", fill=False)
    ax.add_artist(circle)


# 最終軌道の可視化
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
for q in q_traj:
    plot_robot(q, ax)
plt.show()
