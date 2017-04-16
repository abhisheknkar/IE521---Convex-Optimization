import numpy as np
from ellipsoid import readWDBC

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
            print 'Iterations:', iterCount, 'Gradient:',gradNorm, 'Function value:', f_val
    return (w_opt, f_hist)

def gradient_descent_wOffset(X,y,lambda_, w0, gamma, eps):
    gradNorm = np.inf
    w_opt = np.concatenate((w0,[0]))
    f_hist = []

    iterCount = 0

    while gradNorm > eps:
        grad = lambda_* w_opt
        grad[-1] = 0

        f_val = 0.5*lambda_ * (np.linalg.norm(w_opt[:-1]))**2

        for idx,x in enumerate(X):
            EXP_TERM = np.exp(-y[idx]*(w_opt[:-1].dot(x)+w_opt[-1]))
            f_val += np.log(1+EXP_TERM)
            grad[0:-1] += (-y[idx]*EXP_TERM) / (1 + EXP_TERM) * x
            grad[-1] += (-y[idx]*EXP_TERM) / (1 + EXP_TERM)

        f_hist.append(f_val)
        w_opt -= gamma * grad
        gradNorm = np.linalg.norm(grad)

        iterCount += 1
        if iterCount % 100 == 0:
            print 'Iterations:', iterCount, 'Gradient:',gradNorm, 'Function value:', f_val
    return (w_opt, f_hist)


if __name__ == '__main__':
    (X,y) = readWDBC()
    lambda_ = 1.0
    w0 = np.zeros(X.shape[1])
    eps = 1e-10

    [w,v] = np.linalg.eig(X.T.dot(X))
    L = 0.25*max(w) + lambda_
    gamma = 1.0/L

    [w_opt, f_hist] = gradient_descent_wOffset(X, y, lambda_, w0, gamma, eps)