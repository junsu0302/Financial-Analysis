import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
np.random.seed(1000)
#np.set_printoptions(suppress=True, precision=4)

#? 데이터 준비
from sklearn.datasets._samples_generator import make_blobs

X, y = make_blobs(n_samples=250, centers=4, random_state=500, cluster_std=1.25)

# plt.figure(figsize=(10, 6))
# plt.scatter(X[:, 0], X[:, 1], s=50)
# plt.show()

# TODO: k-means 클러스터링
from sklearn.cluster import KMeans

#? k-means 클러스터링 모델 생성 및 학습
model = KMeans(n_clusters=4, random_state=0)
model.fit(X)

#? 예측값 계산
y_kmeans = model.predict(X)

#? 시각화
# plt.figure(figsize=(10, 6))
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='coolwarm')
# plt.show()

#? 데이터 준비
from sklearn.datasets import make_classification

n_samples = 100
X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=250)

# plt.figure(figsize=(10, 6))
# plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap='coolwarm')
# plt.show()

# TODO: 가우스 나이브 베이즈
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#? 가우스 나이브 베이즈 모델 생성 및 학습
model = GaussianNB()
model.fit(X, y)

#? 예측값 및 정확도 계산
pred = model.predict(X)
accuracy = accuracy_score(y, pred)

#? 예측이 맞은 데이터와 틀린 데이터 분리
X_correct = X[y == pred]
y_correct = y[y == pred]
X_incorrect = X[y != pred]
y_incorrect = y[y != pred]

#? 시각화
# plt.figure(figsize=(10, 6))
# plt.scatter(X_correct[:, 0], X_correct[:, 1], c=y_correct, marker='o', cmap='coolwarm', label='Correct') #* 예측이 맞은 데이터 (동그라미로 표시)
# plt.scatter(X_incorrect[:, 0], X_incorrect[:, 1], c=y_incorrect, marker='x', cmap='coolwarm', label='Incorrect') #* 예측이 틀린 데이터 (X자로 표시)
# plt.title(f'Gaussian Naive Bayes Classification Results\nAccuracy: {accuracy:.2f}')
# plt.show()

# TODO: 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

#? 로지스틱 화귀 모델 생성 및 학습
model = LogisticRegression(C=1, solver='lbfgs')
model.fit(X, y)

#? 예측값 및 정확도 계산
pred = model.predict(X)
accuracy = accuracy_score(y, pred)

#? 예측이 맞은 데이터와 틀린 데이터 분리
X_correct = X[y == pred]
y_correct = y[y == pred]
X_incorrect = X[y != pred]
y_incorrect = y[y != pred]

#? 시각화
# plt.figure(figsize=(10, 6))
# plt.scatter(X_correct[:, 0], X_correct[:, 1], c=y_correct, marker='o', cmap='coolwarm', label='Correct') #* 예측이 맞은 데이터 (동그라미로 표시)
# plt.scatter(X_incorrect[:, 0], X_incorrect[:, 1], c=y_incorrect, marker='x', cmap='coolwarm', label='Incorrect') #* 예측이 틀린 데이터 (X자로 표시)
# plt.title(f'Logistic Regression Classification Results\nAccuracy: {accuracy:.2f}')
# plt.show()

# TODO: 의사결정나무
from sklearn.tree import DecisionTreeClassifier

#? 의사결정나무 모델 생성 및 학습
model = DecisionTreeClassifier(max_depth=5)
model.fit(X, y)

#? 예측값 및 정확도 계산
pred = model.predict(X)
accuracy = accuracy_score(y, pred)

#? 예측이 맞은 데이터와 틀린 데이터 분리
X_correct = X[y == pred]
y_correct = y[y == pred]
X_incorrect = X[y != pred]
y_incorrect = y[y != pred]

#? 시각화
# plt.figure(figsize=(10, 6))
# plt.scatter(X_correct[:, 0], X_correct[:, 1], c=y_correct, marker='o', cmap='coolwarm', label='Correct') #* 예측이 맞은 데이터 (동그라미로 표시)
# plt.scatter(X_incorrect[:, 0], X_incorrect[:, 1], c=y_incorrect, marker='x', cmap='coolwarm', label='Incorrect') #* 예측이 틀린 데이터 (X자로 표시)
# plt.title(f'Decision Tree Classification Results\nAccuracy: {accuracy:.2f}')
# plt.show()

# TODO: 소프트 벡터 머신
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#? 데이터를 훈련 세트와 테스트 세트로 분할
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=0)

#? SVC 모델 생성 및 학습
model = SVC(C=1, kernel='linear')
model.fit(train_x, train_y)

#? 예측값 및 정확도 계산
pred = model.predict(test_x)
accuracy = accuracy_score(test_y, pred)

#? 예측이 맞은 데이터와 틀린 데이터 분리
X_correct = test_x[test_y == pred]
y_correct = test_y[test_y == pred]
X_incorrect = test_x[test_y != pred]
y_incorrect = test_y[test_y != pred]

#? 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X_correct[:, 0], X_correct[:, 1], c=y_correct, marker='o', cmap='coolwarm', label='Correct') #* 예측이 맞은 데이터 (동그라미로 표시)
plt.scatter(X_incorrect[:, 0], X_incorrect[:, 1], c=y_incorrect, marker='x', cmap='coolwarm', label='Incorrect') #* 예측이 틀린 데이터 (X자로 표시)
plt.title(f'SVC Classification Results\nAccuracy: {accuracy:.2f}')
plt.show()
