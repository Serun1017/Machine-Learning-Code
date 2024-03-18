from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionGD :
    def __init__(self, eta=0.01, epoch=50, random_state=1) :
        self.eta = eta
        self.epoch = epoch
        self.random_state = random_state

    def fit(self, X, y) :
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses = []

        for i in range(self.epoch) :
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)

            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()

            loss = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) / X.shape[0])
            self.losses.append(loss)

        return self
    
    def net_input(self, X) :
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, z) :
        # 로지스틱 시그모이드 활성화 계산
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X) :
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
# 결정경계 시각화
def plot_decision_regions(X, y, classifier, resolution=0.02) :
    # 마커와 컬러맵 설정
    markers = ('o', 's', '^', 'v', '<')
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정경계를 그림
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)) :
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolors='black')
        
if __name__ == "__main__" :

    ''' dataloader '''
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    print('클래스 레이블: ', np.unique(y))

    # 데이터셋 train, test 분리 및 계층화
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # 데이터셋 계층화 확인
    print('y의 레이블 카운트: ', np.bincount(y))
    print('y_train의 레이블 카운트: ', np.bincount(y_train))
    print('y_test의 레이블 카운트: ', np.bincount(y_test))

    ''' datapreprocessing '''
    # 테스트 데이터셋 표준화
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

    ''' train '''
    lrgd = LogisticRegressionGD(eta=0.3, epoch=10000, random_state=1)
    lrgd.fit(X_train_01_subset, y_train_01_subset)
    
    # decision function visualization
    plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()