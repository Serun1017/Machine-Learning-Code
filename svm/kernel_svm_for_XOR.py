import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn.svm import SVC
from plot_decition_regions import plot_decision_regions

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__' :
    np.random.seed(1)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                           X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, 0)

    plt.scatter(X_xor[y_xor == 1, 0],
                X_xor[y_xor == 1, 1],
                c='royalblue',
                marker='s',
                label='class 1')
    
    plt.scatter(X_xor[y_xor == 0, 0],
                X_xor[y_xor == 0, 1],
                c='tomato',
                marker='o',
                label='class 0')
    
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


    svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)
    
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()