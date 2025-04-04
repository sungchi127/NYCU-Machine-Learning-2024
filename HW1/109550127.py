import numpy as np
import matplotlib.pyplot as plt

# shape(23, 1) to (1, 23)
def shape_tran(old):
    tmp = np.zeros((len(old[0]),len(old)))
    for r in range(len(old)):
        for c in range(len(old[r])):
            tmp[c][r] = old[r][c]
    return tmp

#計算x^0、x^1、和x^2的值，並將這些值存儲在矩陣的對應行中
def PolyM(A,N):
    tmp_A = np.zeros((len(A[0]),N))
    for r in range(len(tmp_A)):
        for i in range(N):
            tmp_A[r][N-1-i] = A[0][r]**i
    return tmp_A

#矩陣乘法
def multi(a,b): 
    # print(a.shape, b.shape)
    if len(a[0])!=len(b):
        print('wrong size of matrix in multiplication')
        return
    multmp = np.zeros((len(a),len(b[0])))
    for i in range(len(multmp)):
        for j in range(len(multmp[i])):
            for k in range(len(a[0])):
                multmp[i][j] += a[i][k]*b[k][j]
    return multmp

#常數乘法
def Cmulti(c, ma):
    Ctmp = ma.copy()
    Ctmp[:][:] = c * ma[:][:] 
    return Ctmp  
  
#lambda對角矩陣
def unitM(length, _lambda=1):
    unit = np.zeros((length,length))
    for i in range(length):
        unit[i][i] = 1*_lambda
    return unit

#矩陣加法
def add(a, b, minus=False):
    if len(a)!=len(b) or len(a[0])!=len(b[0]):
        print('wrong size of matrix in addition')
        return
    tmp = np.zeros((len(a),len(a[0])))
    for i in range(len(a)):
        for j in range(len(a[i])):
            tmp[i][j] = a[i][j] + b[i][j] if not minus else a[i][j] - b[i][j]
    return tmp

#LU分解求A的inverse => A^-1=U^-1*L^-1
def LUdecomp(A):
    n = len(A)
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            sum1 = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum1
        for j in range(i + 1, n):
            sum2 = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - sum2) / U[i][i]

    # 用前向替換求解L * Y = I求L的逆
    Linv = np.zeros_like(L)
    for i in range(n):
        Linv[i][i] = 1 / L[i][i]
        for j in range(i + 1, n):
            Linv[j][i] = -sum(L[j][k] * Linv[k][i] for k in range(i, j)) / L[j][j]

    # 用后向替換求解U * X = I求U的逆
    Uinv = np.zeros_like(U)
    for i in reversed(range(n)):
        Uinv[i][i] = 1 / U[i][i]
        for j in reversed(range(i)):
            Uinv[j][i] = -sum(U[j][k] * Uinv[k][i] for k in range(j + 1, i + 1)) / U[j][j]

    # 計算A的逆为U的逆乘以L的逆
    Ainv = multi(Uinv, Linv)
    return Ainv

def steepest_descent_L1(AP, b, N, learning_rate, iterations):
    m, n = AP.shape  # m是數據點数量, n是特徵數量
    # 23*3
    # print(AP)
    # print(m,n)
    X = np.zeros((n, 1))  # 初始化X為n*1的向量

    for it in range(iterations):
        gradient = np.zeros((n, 1))  # 初始化梯度为n*1的向量
        # 計算L1的梯度 ∑i=1->m ∣APi*​X−b∣
        for i in range(m):
            prediction = np.dot(AP[i], X)
            residual = prediction - b[i]
            for j in range(n):
                gradient[j] += np.sign(residual) * AP[i, j]

        # 更新X
        X = X - learning_rate * gradient

        # 計算當前的誤差
        err = np.sum(np.abs(multi(AP, X) - b))

        # if it % 100 == 0:
            # print(f'Iteration {it+1}, Error: {err}')
    return X



def Fitting_and_error(A, b, X, N, _lambda):
    # Calculate error
    err = 0.0
    for i in range(len(A)):
        pre = 0.0
        for n in range(N):
            pre += A[i][n] * X[n]
        err += (pre - b[i])**2
    for n in range(N):
        err += _lambda * (X[n]**2)
    
    # Output fitting line and total error
    fitting_line = '\tFitting line:\n\t\t'
    for n in range(N):
        fitting_line += '{:.10f}'.format(X[n][0])
        if n != N-1:
            fitting_line += ' X^{} + '.format(N-1-n)
    total_error = '\n\tTotal error: {}'.format(err[0])
    
    # Combine fitting line and total error in one print statement
    print(fitting_line + total_error)
    return err


# text_name = input('Enter the name of the file: ')
text_name = 'data.txt'
N = int(input('Enter the number of polynomial bases n: '))
# N = 3
_lambda = float(input('Enter the value of lambda: '))
# _lambda = 0
file = open(text_name,'r')
data = []
for line in file.readlines():
    tmp = line.split(',')
    tmp = [float(t) for t in tmp]
    # print(tmp,tmp[0],tmp[1])
    data.append(tmp)
A = [[ A[0] for A in data]]
A = np.array(A)
B = [[ B[1] for B in data]]
B = np.array(B)

B = shape_tran(B)

#RLSE
# X = (ATA + lambda*I)^-1 * AT * b
AP = PolyM(A,N)
# print(A)
# print(AP)
APT = shape_tran(AP)
X = add(multi(APT, AP),unitM(N,_lambda))
X = LUdecomp(X)
X = multi(multi(X, APT), B)

print('LSE:')
Fitting_and_error(AP, B, X, N, _lambda)


#Steepest Descent
X2 = steepest_descent_L1(AP, B, N, 0.001, 15000)

print("Steepest Descent:")
Fitting_and_error(AP, B, X2, N, 0)


#Newton's Method
# L = ||AX-b||^2
X3 = np.zeros((N,1))
tmp1 = Cmulti(2, multi(multi(APT, AP), X3))
tmp2 = Cmulti(2, multi(APT, B))
delf = add(tmp1, tmp2, True) # 2*AT*A*X−2*AT*B
Hfinv = LUdecomp(Cmulti(2, multi(APT, AP)))#2*AT*A
X3 = add(X3, multi(Hfinv, delf), True)

print("Newton's Method:")
Fitting_and_error(AP, B, X3, N, 0)


#Visualize
# LSE
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax1.scatter(A[0], B, color='red')
x_line = np.linspace(-6, 6, 30000)  # 生成線的採樣點
y_line = np.zeros(len(x_line))
for i in range(len(x_line)):
    for n in range(N):
        y_line[i] += X[n] * (x_line[i]**(N-n-1))
ax1.plot(x_line, y_line, 'k')

#Steepest Descent
ax2 = fig.add_subplot(3, 1, 2)
ax2.scatter(A[0], B, color='red')
y_line = np.zeros(len(x_line))  # 重新計算y_line以便繪製新的線
for i in range(len(x_line)):
    for n in range(N):
        y_line[i] += X2[n] * (x_line[i]**(N-n-1))
ax2.plot(x_line, y_line, 'k')

# Newton's Method
ax3 = fig.add_subplot(3, 1, 3)
ax3.scatter(A[0], B, color='red')
y_line = np.zeros(len(x_line))  # 重新計算y_line以便繪製新的線
for i in range(len(x_line)):
    for n in range(N):
        y_line[i] += X3[n] * (x_line[i]**(N-n-1))
ax3.plot(x_line, y_line, 'k')

ax1.set_title('RLSE vs Steepest Descent vs Newton\'s Method')
plt.show()

