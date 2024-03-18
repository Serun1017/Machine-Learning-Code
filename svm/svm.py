import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from plot_decition_regions import plot_decision_regions

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__' :
    
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

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # SVM model
    # svm = SVC(kernel='linear', C=1.0, random_state=1)

    # kernel SVM model
    # svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=10.0)
    # svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
    svm = SVC(kernel='rbf', random_state=1, gamma=100, C=1.0)

    svm.fit(X_train_std, y_train)
    
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()