import random
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

'''
1. ガウス分布から長さLのz(t)ベクトルを生成
2. 平均ベクトルμ(既知)，共分散行列Σ(既知)とz(t)からy(t)を作成
3. y(t)からx(t)を作成
パラメータはαだけ
'''

network_mat = [[0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,],
               [1, 0, 2, 2, 2, 2, 1, 1, 3, 3, 3, 3,],
               [0, 0, 0, 2, 2, 2, 3, 3, 1, 1, 3, 3,],
               [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 1, 1,],
               [0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 3, 3,],
               [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3,],
               [0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 4, 4,],
               [0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4,],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4,],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4,],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]]

def hop(i, j):
    a = network_mat[i][j]
    b = network_mat[j][i]
    return max(a, b)


# 初期化
L = 12
N = 1000
sigma = 0.000000005
c = 0.1
mu_l = 500

alpha = 0.75
delta = 1

mu_0 = np.zeros(L)
cov_matrix_0 = np.matrix(np.eye(L)) * (sigma ** 2)

# 平均ベクトル
mu = np.matrix(np.array([mu_l for i in range(L)])).T

# 分散共分散行列
cov_matrix = np.matrix(np.zeros((L, L)))
for i in range(L):
    for j in range(L):
        if i == j:
            cov_matrix[i, j] = (c * mu[i]) ** 2
        else:
            s_ii = (c * mu[i]) ** 2
            s_jj = (c * mu[j]) ** 2
            cov_matrix[i, j] = math.sqrt(s_ii * s_jj) * math.e ** (-hop(i, j) / delta)

R = np.linalg.cholesky(cov_matrix)

x_list = []
for t in range(1, N+1):
    z = np.matrix(np.random.multivariate_normal(np.zeros(L), cov_matrix_0, 1)).T
    y = R * z + mu
    if t == 1:
        x_list.append(y)
    else:
        print(t)
        print(x_list[t-1 - 1])
        print(y)
        val = alpha * x_list[t-1 - 1] + (1-alpha) * y
        x_list.append(val)

X = np.matrix(np.array(x_list)).T


# 課題1
# print(X.shape)
# for i in range(X.shape[1]):
#     print(np.array(X[:,i]))
#     plt.plot(np.array(X[:,i]))
#
# plt.ylim(500-10**(-6), 500+10**(-6))
# plt.show()

# 課題2
# mu_hat = np.mean(X, axis=1)
# one_vec = np.matrix(np.ones(N))
# sigma_hat = (1 / N) * (X - (mu_hat * one_vec)) * (X - (mu_hat * one_vec)).T
#
# eig_val, eig_vec = np.linalg.eig(sigma_hat)
# print(eig_val)
# plt.plot(eig_val, label=str(delta))
#
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("eigenvalue")
# plt.show()

# 課題3
beta = 0.5
K = 3
r = 5
# 異常トラヒックの追加
for i in range(K):
    t = random.randint(0, N-1)
    l = random.randint(0, L-1)
    X[l, t] = X[l, t] + 500 + 500 * c * beta
    print(l, t)
mu_hat = np.mean(X, axis=1)
one_vec = np.matrix(np.ones(N))
sigma_hat = (1 / N) * (X - (mu_hat * one_vec)) * (X - (mu_hat * one_vec)).T

eig_val, eig_vec = np.linalg.eig(sigma_hat)
eig_vec_0 = eig_vec[:, 0:r+1]

X_pca = eig_vec_0.T * X

plt.plot(X_pca.T)
plt.legend()
plt.show()
