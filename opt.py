import casadi as ca
import numpy as np

# 時間ステップ
T = 0.1
N = 20

# シンボリック変数の定義
x = ca.MX.sym('x')  # 状態変数
u = ca.MX.sym('u')  # 制御変数

# 運動モデルの定義
x_next = x + T*u

# オプティミゼーション変数の定義
opti = ca.Opti()
X = opti.variable(N+1)  # 状態変数の軌跡
U = opti.variable(N)    # 制御変数の軌跡

# 初期条件
x0 = 0
opti.subject_to(X[0] == x0)

# 目標状態
x_target = 1

# 目的関数と制約条件の定義
J = 0
for k in range(N):
    J += (X[k] - x_target)**2  # 目的関数: 状態の偏差の二乗和
    opti.subject_to(X[k+1] == X[k] + T*U[k])  # システムダイナミクス
    opti.subject_to(U[k] <= 1)  # 制御入力の上限
    opti.subject_to(U[k] >= -1) # 制御入力の下限

opti.minimize(J)

# ソルバーの設定と最適化の実行
opti.solver('ipopt')
sol = opti.solve()

# 結果の取得
x_opt = sol.value(X)
u_opt = sol.value(U)

# 結果の表示
print("Optimal state trajectory:", x_opt)
print("Optimal control trajectory:", u_opt)

# 結果のプロット
import matplotlib.pyplot as plt 
plt.figure()
plt.plot(x_opt, label='x')
plt.plot(u_opt, label='u')

plt.xlabel('Time step')

plt.legend()
plt.show()
