from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt

# 결정경계 시각화
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02) :
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

    # 테스트 샘플을 부각함
    if test_idx :
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    alpha=1.0,
                    edgecolors='black',
                    linewidths=1,
                    marker='o',
                    s=100,
                    label='Test set')
    
if __name__ == '__main__': 

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

    ''' train '''
    # 퍼셉트론 학습
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)
    print('잘못 분류된 샘플 새수: %d' % (y_test != y_pred).sum())

    print('정확도: %.3f' %accuracy_score(y_test, y_pred))

    # 결정경계 시각화
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))

    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()