import numpy as np
import matplotlib.pyplot as plt
MAXITERS = 1000

def readWDBC():
    f = open('wdbc.data','r')
    X = []
    Y = []
    labelMap = {'B':1.0,'M':-1.0}
    for line in f.readlines():
        lsplit = line.strip().split(',')
        Y.append(labelMap[lsplit[1]])
        toAdd = []
        for x in lsplit[2:]:
            toAdd.append(float(x))
        X.append(toAdd)
    return np.array(X), np.reshape(np.array(Y),(len(Y),1))

def gradient_descent(X,y,lambda_, w0, gamma, eps):
    gradNorm = np.inf
    w_opt = w0
    f_hist = []

    iterCount = 0

    while gradNorm > eps:
        grad = lambda_*w_opt
        f_val = 0.5*lambda_ * (np.linalg.norm(w_opt))**2

        for idx,x in enumerate(X):
            EXP_TERM = np.exp(-y[idx]*w_opt.dot(x))
            f_val += np.log(1+EXP_TERM)
            grad += (-y[idx]*EXP_TERM) / (1 + EXP_TERM) * x

        f_hist.append(f_val)
        w_opt -= gamma * grad
        gradNorm = np.linalg.norm(grad)

        iterCount += 1
        if iterCount % 100 == 0:
            print 'Iterations:', iterCount, 'Gradient Norm:',gradNorm, 'Function value:', f_val
        if iterCount > MAXITERS:
            break
    return (w_opt, f_hist)

def Newton(X, y, lambda_, w0, eps):
    w_opt = w0
    f_hist = []
    grad = lambda_ * w_opt
    gradNorm = np.inf
    iterCount = 0

    while gradNorm > eps:
        grad = lambda_*w_opt
        hessian = lambda_ * np.identity(len(w0), dtype=float)
        f_val = 0.5*lambda_ * (np.linalg.norm(w_opt))**2

        for idx,x in enumerate(X):
            EXP_TERM = np.exp(-y[idx]*w_opt.dot(x))
            hessian += EXP_TERM / ((1 + EXP_TERM)**2) * np.outer(x,x)
            f_val += np.log(1+EXP_TERM)
            grad += (-y[idx]*EXP_TERM) / (1 + EXP_TERM) * x
        f_hist.append(f_val)
        inv_hessian = np.linalg.inv(hessian)

        # plt.imshow(inv_hessian, interpolation='nearest', cmap=plt.cm.ocean,
        #            extent=(0.5, np.shape(inv_hessian)[0] + 0.5, 0.5, np.shape(inv_hessian)[1] + 0.5))
        # plt.colorbar()
        # plt.show()

        w_opt -= np.linalg.inv(hessian).dot(grad)
        gradNorm = np.linalg.norm(grad)
        iterCount += 1
        if iterCount % 1 == 0:
            print 'Iterations:', iterCount, 'Gradient norm:',gradNorm, 'Function value:', f_val
        if iterCount > MAXITERS:
            break
    return (w_opt, f_hist)

def damped_Newton(A, b, c, x0, eps):
    x_opt = x0
    lambda_hist = []
    f_hist = []
    lambda_ = np.infty

    while lambda_ > eps:
        f = c.T.dot(x_opt)
        grad = c.copy()
        hessian = np.zeros((len(x0),len(x0)), dtype=float)

        # Get the gradient, hessian
        for idx,row in enumerate(A):
            ERROR = b[idx]-row.dot(x_opt)
            f -= np.log(ERROR)
            grad += row / ERROR
            hessian += np.outer(row, row) / (ERROR*ERROR)

        inv_hessian = np.linalg.inv(hessian)
        lambda_ = np.sqrt(grad.dot(inv_hessian).dot(grad))
        x_opt -= inv_hessian.dot(grad) / (1+lambda_)
        f_hist.append(f)
        lambda_hist.append(lambda_)
        print 'Newton step:',lambda_
    return (x_opt, f_hist, lambda_hist)

def Q1Exp():
    toDos = [2]
    (X,y) = readWDBC()
    lambda_ = 1.0
    w0 = np.zeros(X.shape[1])
    eps = 1e-3

    [w,v] = np.linalg.eig(X.T.dot(X))
    L = 0.25*max(w) + lambda_
    gamma = 1.0/L

    # 1.3, part 1
    if 1 in toDos:
        stepSizes = [x/L for x in [1e-4,1e-3,1e-2,1e-1,1]]
        for x in stepSizes:
            [w_opt, f_hist] = gradient_descent(X, y, lambda_, w0, x, eps)
            plt.plot(f_hist, label='Stepsize:'+str(x))
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function')
        plt.title('Variation of objective function trajectory with step size using gradient descent')
        plt.legend()
        plt.show()

    # 1.3, part 2
    if 2 in toDos:
        meanList = [0]
        stdList = [1e-5,1e-3, 1e-1]
        for m in meanList:
            for x in stdList:
                print 'Mean:',m,'STD:', x
                w0 = np.random.normal(loc=m, scale=x, size=(X.shape[1]))
                [w_opt, f_hist] = Newton(X, y, lambda_, w0, eps)
                plt.plot(f_hist, label='Mean:'+str(m)+', '+'Standard Deviation:'+str(x))
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function')
        plt.title('Variation of objective function trajectory with standard deviation using Newton Method')
        plt.legend()
        plt.show()

    if 3 in toDos:
        w0 = np.zeros(X.shape[1])
        [w_opt, f_hist1] = gradient_descent(X, y, lambda_, w0, gamma, eps)
        plt.plot(f_hist1, label='Gradient Descent')

        [w_opt, f_hist2] = Newton(X, y, lambda_, w0, eps)
        f_hist2 += [f_hist2[-1] for x in range(len(f_hist1)-len(f_hist2))]
        plt.plot(f_hist2, label='Newton Method')
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function')
        plt.title('Comparison of Gradient Descent with Newton Method')
        plt.legend()
        plt.show()

def Q2Exp():
    m,n = 100,50
    eps = 1e-10
    A = np.random.normal(loc=0.0, scale=1.0, size=(m,n))
    b = np.random.uniform(low=0.0, high=1.0, size=m)
    c = np.random.normal(loc=0.0, scale=1.0, size=n)
    x0 = np.zeros(n, dtype=float)
    (x_opt, f_hist, lambda_hist) = damped_Newton(A, b, c, x0, eps)

    f_opt = min(f_hist)

    plt.plot([x-f_opt for x in f_hist])
    plt.xlabel('Iterations')
    plt.ylabel('Approximate function gap')
    plt.title('Variation of function gap over iterations')
    plt.show()

    plt.plot(lambda_hist)
    plt.xlabel('Iterations')
    plt.ylabel('Newton step')
    plt.title('Variation of Newton step over iterations')
    plt.show()

if __name__ == '__main__':
    Q1Exp()
    Q2Exp()
