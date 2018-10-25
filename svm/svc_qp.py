import numpy as np
import cvxopt
import cvxopt.solvers
import svm.kernels as kernels


class SVC:
    """
    Suppoet vector classification by quadratic programming
    """
    def __init__(self, kernel="linear", C=None):
        """
        :param kernel: kernel types, should be in the kernel function list above
        :param C:
        """
        self.kernel = kernels.KERNEL_TYPES[kernel]
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = self.kernel(X, X)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G1 = np.identity(n_samples) * -1
            G2 = np.identity(n_samples)
            h1 = np.zeros(n_samples)
            h2 = np.ones(n_samples) * self.C
            G = cvxopt.matrix(np.vstack((G1, G2)))
            h = cvxopt.matrix(np.hstack((h1, h2)))

        # solve QP problem, DOC: http://cvxopt.org/userguide/coneprog.html?highlight=qp#cvxopt.solvers.qp
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # 这里计算b其实是错误的，只能通过free support vector
        # 计算b, 不能用bounded support vector
        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == kernels.linear_kernel:
            self.w = np.dot(self.a * self.sv_y, self.sv)
            # self.w = np.zeros(n_features)
            # for n in range(len(self.a)):
            #     self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            # y_predict = np.zeros(len(X))
            # for i in range(len(X)):
            #     s = 0
            #     for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
            #         s += a * sv_y * self.kernel(sv, X[i])
            #     y_predict[i] = s
            y_predict = np.sum(self.a * self.sv_y * self.kernel(X, self.sv), axis=1) +self.b
            return y_predict

    def predict(self, X):
        pred_y = np.sign(self.project(X))
        pred_y[pred_y == 0] = 1

        return pred_y

    def score(self, X_test, y_test):
        y = self.predict(X_test)
        return np.sum(y == y_test) / y_test.shape[0]