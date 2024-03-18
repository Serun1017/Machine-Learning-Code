import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

class AdalinGD :
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
        losses : list
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
        self.losses_ = []

        for _ in range(self.n_iter) :
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self
        
    def net_input(self, X) :
        """ 최종 입력 계산 """
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X) :
        """ 선형 활성화 계산"""
        return X
    
    def predict(self, X) :
        """ 계단 함수를 사용하여 클래스 레이블을 반환한다. """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

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
    # 표준화(standardization)
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    
    # perceptron 생성
    ppn = AdalinGD(eta=0.1, n_iter=20)
    # perceptron 훈련
    ppn.fit(X_std, y)

    # visualization
    plt.plot(range(1, len(ppn.losses_) + 1), ppn.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

    print(ppn.losses_)