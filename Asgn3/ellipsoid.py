import matplotlib.pyplot as plt
import numpy as np

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

def createDummyDataset(n=1000, rotation=0, plotFlag=False, mode='2'):
    if mode == '2':
        sigma = 0.1
        s1 = np.random.normal(1, sigma, n)
        s2 = np.random.normal(-1, sigma, n)
        s0 = np.random.normal(0,sigma,2*n)

        class1 = [(x,y) for x,y in zip(s1,s0[0:n])]
        class2 = [(x,y) for x,y in zip(s2,s0[n:])]

        X1 = np.array(class1)
        X2 = np.array(class2)
        X = np.vstack((X1,X2))
        Y = np.vstack((np.ones((n,1),dtype=np.int),-np.ones((n,1),dtype=np.int)))

        # Rotate
        rotationMatrix = np.array([[np.cos(rotation),-np.sin(rotation)],[np.sin(rotation),np.cos(rotation)]])
        for i in range(X.shape[0]):
            X[i,:] = np.dot(rotationMatrix,X[i,:])

    elif mode == '1':
        sigma = 0.1
        s1 = np.random.normal(3, sigma, n)
        s2 = np.random.normal(1, sigma, n)

        X1 = np.array(s1)
        X2 = np.array(s2)
        X = np.vstack((X1, X2))
        Y = np.vstack((np.ones((n, 1), dtype=np.int), -np.ones((n, 1), dtype=np.int)))

    return (X,Y)

def ellipsoid(X,Y, lam=1, nmax=100, plotFlag=True):
    # Initialize the following: parameters - w,b; subgradient - omega;
    # incorrect classifications - wrong; ellipse - Q, mat
    n = X.shape[1]
    m = X.shape[0]
    c = np.zeros((n+1,1))
    omega = np.zeros(c.shape)   # Sub-gradient

    lossPlot = []
    minlossPlot = []

    R = 5
    Q = np.identity(n+1)*(R*R)    # Ellipsoid
    for i in range(nmax):
        classifications = 1 - Y*(X.dot(c[0:n])+c[n])   # 1 - y.*(w^T x + b)
        wrong = classifications > 0
        loss = 1.0/m*sum(wrong*classifications) + lam*c[:n].T.dot(c[:n])
        lossPlot.append(float(loss))
        minlossPlot.append(min(lossPlot))

        # if i%10 == 0:
        #     print i, loss

        sum_xy = np.zeros((n,1))
        for k in range(m):
            sum_xy += np.reshape(Y[k]*X[k,:]*wrong[k],(n,1))
        omega[:n] = (2*lam*c[:n] - sum_xy )/ m
        omega[n] = -np.sum(Y*wrong)/m

        # Update c, Q
        c = c - 1.0 / ( (n+1) * np.sqrt( omega.T.dot(Q).dot(omega) +1e-10) ) * Q.dot(omega)
        Q = float(n*n)/(n*n-1) * (Q - 2.0/(n+1) * Q.dot(omega).dot(omega.T).dot(Q) / ( omega.T.dot(Q).dot(omega)  + 1e-10 ))

    if plotFlag:
        if n == 2:
            xdata = X[:,0]
            ydata = X[:,1]
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)

            ax1.scatter(xdata, ydata)
            (x_,y_) = graphCoordinates(c[:n],c[n],2)
            ax1.plot(x_,y_)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        ax2.plot(range(0,len(lossPlot)),lossPlot, label='Objective Function')
        ax2.plot(range(0,len(lossPlot)),minlossPlot, label='Cumulative Minimum of Objective Function')

        plt.xlabel('Timestep')
        plt.ylabel('Objective Function')
        plt.title('Variation of Objective Function Over Time')
        plt.legend()
        plt.show()

    print 'Misclassification Error:', float(sum(wrong)) / m

def graphCoordinates(w,b,range_x):
    w_ = [float(i) for i in w]
    b_ = float(b)

    x = range(-range_x,range_x+1)
    y = [1.0/w_[1]*(-w_[0]*i + b_) for i in x]
    return (x,y)

if __name__ == '__main__':
    # (X,Y) = createDummyDataset(rotation=1)
    (X,Y) = readWDBC()
    ellipsoid(X,Y)
