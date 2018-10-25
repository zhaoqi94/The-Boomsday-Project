import numpy as np
import svm.kernels as kernels

# binary svm
class SVC:
    def __init__(self,C=1.0, kernel='linear',max_iter=-1, tol=1e-3, eps=1e-4):
        self.b = 0.0
        self.C = C
        self.kernel = kernels.KERNEL_TYPES[kernel]
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

        # 未定的参数
        # fit的时候才确定
        self.input_size = -1
        self.X = None
        self.y = None
        self.alpha = None
        self.support_vector = None
        self.support_vector_label = None
        self.support_vector_alpha = None
        # error的缓存，不需要限制大小
        self.error_cache = None
        # 核矩阵的缓存，加快训练速度
        # 但是当样本数很多时，内存会不够用！
        # 所以需要学习libsvm的缓存机制，去限制最大的大小
        self.kernel_cache = None

    def fit(self, X, y):
        # 初始化一下参数
        self.X = X
        self.y = y
        self.input_size = X.shape[0]
        self.error_cache = np.zeros(self.input_size) - y
        self.kernel_cache = self.kernel(X, X)
        self.alpha = np.zeros(self.input_size)
        self.b = 0.0

        num_changed = 0
        iter_count = 0
        examine_all = 1

        # outer loop
        while ((iter_count<self.max_iter) and (num_changed > 0 or examine_all)):
            print(iter_count)
            num_changed = 0
            if examine_all:
                for i in range(self.input_size):
                    num_changed += self.examine_example(i)
            else:
                for i in range(self.input_size):
                    if (self.alpha[i] != 0 and self.alpha[i] != self.C):
                        num_changed += self.examine_example(i)
            if examine_all:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

            iter_count += 1
            print(num_changed)

        #record support vector and corresponding alpha
        support_vector_indices = self.alpha != 0.0
        self.support_vector_alpha = self.alpha[support_vector_indices]
        self.support_vector = self.X[support_vector_indices]
        self.support_vector_label = self.y[support_vector_indices]

    def predict(self, X):
        # kernel(data, self.support_vector)
        # --> [input_size, support_vector_num]
        y_pred = np.sum(self.support_vector_alpha * self.support_vector_label
                   * self.kernel(X, self.support_vector), axis=1) + self.b
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / X.shape[0]

    def examine_example(self, i1):
        y1 = self.y[i1]
        alpha1 = self.alpha[i1]
        E1 = self.error_cache[i1]
        r1 = E1 * y1

        # 检查KKT条件
        # alpha=0 ==> y*f(x)>1
        # alpha=C ==> y*f(x)<1
        # 0<alpha<C ==> y*f(x)=1
        if (r1 < -self.tol and alpha1 < self.C) or (r1 > self.tol and alpha1 > 0):
            #按照KKT三个KKT条件的重要性选一个alpha进行优化
            #如果还有非边界的alpha，那么就优先选取非边界的alpha
            #优先选择abs(E1-E2)最大的点
            # i2 = self.selectMaxJ(E1)
            if len(self.alpha[(self.alpha != 0) & (self.alpha != self.C)]) > 1:
                i2 = 0
                if self.error_cache[i1] > 0:
                    i2 = np.argmin(self.error_cache)
                else:
                    i2 = np.argmax(self.error_cache)
                if self.take_step(i2, i1):
                        return True

            # 如果abs(E1-E2)最大的点优化失败，则遍历非边界的alpha(即free support vector)，进行优化
            for i2 in np.roll(np.where((self.alpha != 0) & (self.alpha != self.C))[0],
                              np.random.choice(np.arange(self.input_size))):
                if self.take_step(i2, i1):
                    return True

            # 如果非边界的alpha都不需要更新的话，则遍历整个数据集
            for i2 in np.roll(np.arange(self.input_size), np.random.choice(np.arange(self.input_size))):
                if self.take_step(i2, i1):
                    return True

        return False

    def take_step(self, i1, i2):
        if i1 == i2:
            return False
        alpha1 = self.alpha[i1]
        alpha2 = self.alpha[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.error_cache[i1]
        E2 = self.error_cache[i2]
        s = y1 * y2

        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        if L >= H:
            return False

        # k11 = self.kernel(self.X[i1], self.X[i1])
        # k12 = self.kernel(self.X[i1], self.X[i2])
        # k22 = self.kernel(self.X[i2], self.X[i2])
        k11 = self.kernel_cache[i1, i1]
        k12 = self.kernel_cache[i1, i2]
        k22 = self.kernel_cache[i2, i2]

        eta = k11 + k22 - 2*k12
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            print('eta=%f' % eta)
            C1 = eta / 2
            C2 = y2 * (E1 - E2) + eta * alpha2
            Lobj = C1 * L * L + C2 * L
            Hobj = C1 * H * H + C2 * H
            if (Lobj > Hobj + self.eps):
                a2 = L
            elif (Lobj < Hobj - self.eps):
                a2 = H
            else:
                a2 = alpha2

        # Push a2 to 0 or C if very close
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (self.C - 1e-8):
            a2 = self.C

        # 相对误差
        if abs(a2-alpha2) < self.eps*(a2+alpha2+self.eps):
            return False

        a1 = alpha1 + s * (alpha2 - a2)
        if a1 < 0:
            a2 += s * a1
            a1 = 0
        elif a1 > self.C:
            a2 += s * (a1 - self.C)
            a1 = self.C

        # update threshold
        b1 = self.b - E1 - y1 * (a1 - alpha1) * k11 - y2 * (a2 - alpha2) * k12
        b2 = self.b - E2 - y1 * (a1 - alpha1) * k12 - y2 * (a2 - alpha2) * k22
        if (0 < a1 and a1 < self.C):
            bNew = b1
        elif (0 < a2 and a2 < self.C):
            bNew = b2
        else:
            bNew = (b1 + b2) / 2

        self.alpha[i1] = a1
        self.alpha[i2] = a2

        # Update error cache
        # Error cache for optimized alphas is set to 0 if they're unbound
        for index, alpha in zip([i1, i2], [a1, a2]):
            if 0.0 < alpha < self.C:
                self.error_cache[index] = 0.0
            # 奇了怪了！为什么去掉这个else效果反而更好了
            # else:
            #     self.error_cache[index] = self.error_cache[index] + \
            #                                 y1 * (a1 - alpha1) * self.kernel(self.X[i1], self.X[index]) + \
            #                                 y2 * (a2 - alpha2) * self.kernel(self.X[i2], self.X[index]) \
            #                                 - self.b + bNew
        # Set non-optimized errors based on equation 12.11 in Platt's book
        # non_opt = [n for n in range(self.input_size) if (n != i1 and n != i2)]
        # self.error_cache[non_opt] = self.error_cache[non_opt] + \
        #                                     y1 * (a1 - alpha1) * self.kernel(self.X[i1], self.X[non_opt]) + \
        #                                     y2 * (a2 - alpha2) * self.kernel(self.X[i2], self.X[non_opt]) \
        #                                     - self.b + bNew
        non_opt = [n for n in range(self.input_size) if (n != i1 and n != i2)]
        self.error_cache[non_opt] = self.error_cache[non_opt] + \
                                            y1 * (a1 - alpha1) * self.kernel_cache[i1, non_opt] + \
                                            y2 * (a2 - alpha2) * self.kernel_cache[i2, non_opt] \
                                            - self.b + bNew

        self.b = bNew

        return True

    # def decision_function(self, x):
    #     sum = 0.0
    #     for i in range(self.input_size):
    #         if self.alpha[i] != 0:
    #             sum += self.alpha[i]*self.y[i]*self.kernel(self.X[i], x)
    #     sum += self.b
    #     return sum

    ## 定义目标函数
    # def objective_function(self, alphas, target, kernel, X_train):
    #     return np.sum(alphas) - 0.5 * np.sum(target * target * kernel(X_train, X_train) * alphas * alphas)

    # def calc_error(self, k):
    #     sum = 0.0
    #     for i in range(self.input_size):
    #         if self.alpha[i] != 0:
    #             sum += self.alpha[i]*self.y[i]*self.kernel(self.X[k], self.X[i])
    #     sum += self.b
    #     sum -= self.y[k]
    #     return sum

