import casadi as ca

x1 = ca.SX.sym("x1")
x2 = ca.SX.sym("x2")


def f_(x1, x2):
    return x1**2 + x2**2


# x1^2 + x2^2
f = f_(x1, x2)

# 制約条件，x1 * x2 >= 1,  4 >= x1, x2 >= 0
g1 = x1 * x2 - 1
g2 = x1
g3 = x2

# 制約条件をベクトルにまとめる
g = ca.vertcat(g1, g2, g3)

# 最適化する
nlp = {"x": ca.vertcat(x1, x2), "f": f, "g": g}

solver = ca.nlpsol("solver", "ipopt", nlp)
lbg = [0, 0, 0]
ubg = [ca.inf, 4, 4]
res = solver(x0=[-30, -3000], lbg=lbg, ubg=ubg)

res_opt = res["x"]

print(res_opt)  # [1, 1]が最適解
