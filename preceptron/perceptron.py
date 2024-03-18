import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron :
    """ 퍼셉트론 분류기
    매개변수
        eta : float
            학습률 (0.0과 1.0 사이)
        n_iter : int
            훈련 데이터셋 반복 횟수 (epoch)
        random_state : int
            가중치 무작위 초기화를 위한 난수 생성기 시드
    속성
        w_ : 1d-array
            학습된 가중치
        b_ : 스칼라
            학습된 bias 유닛
        errors_ : list
            에포크마다 누적된 분류 오류
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1) :
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y) :
        """ 훈련 데이터 학습
        매개 변수
            X : {array-like}, shape = [n_samples, n_features]
                n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
            y : array-like, shape = [n_samples]
                타깃 값
        반환값
            self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = [] 

        for _ in range(self.n_iter) :
            errors = 0
            for xi, target in zip(X, y) :
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update!= 0.0)
            self.errors_.append(errors)
        return self
            

    def net_input(self, X) :
        """입력 계산"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X) :
        """ 계단 함수를 사용하여 클래스 레이블을 반환한다. """
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
def plot_decision_regions(X, y, classifier, resolution=0.02) :
    # 마커와 컬러맵을 설정
    markers = ('o', 's', '^', 'v', '<')
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정경계를 그림
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 샘플의 산전도를 그림
    for idx, cl in enumerate(np.unique(y)) :
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

if __name__ == "__main__" :

    # dataloader
    s = 'https://archive.ics.uci.edu/ml/'\
        'machine-learning-databases/iris/iris.data'
    print('URL:', s)

    df = pd.read_csv(s, header=None, encoding='utf-8')
    df.tail()


    # setosa와 versicolor를 선택한다.
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    # 꽃받침 길이와 꽃잎 길이를 추출
    X = df.iloc[0:100, [0, 2]].values
    # 산점도를 그린다.
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], 
                color='blue', marker='s', label='Versicolor')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')

    plt.show()
    
    # perceptron 생성
    ppn = Perceptron(eta=0.1, n_iter=10)
    # perceptron 훈련
    ppn.fit(X, y)

    # visualization
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

    # 결정경계 시각화
    plot_decision_regions(X=X, y=y, classifier=ppn)
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

    print(ppn.errors_)