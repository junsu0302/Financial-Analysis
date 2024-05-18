import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

#! 근사화의 핵심은 회귀법과 보간법을 활용하는 것이다.

def f(x): #? 근사할 목표 함수
  return np.sin(x) + 0.5 * x

def create_plot(x, y, styles, labels, axlabels):
  plt.figure(figsize=(10, 6))
  for i in range(len(x)):
    plt.plot(x[i], y[i], styles[i], label=labels[i])
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1])
  plt.legend(loc=0)

x = np.linspace(-2 * np.pi, 2 * np.pi, 50)

# create_plot([x], [f(x)], ['b'], ['f(x)'], ['x', 'f(x)'])

# TODO: 회귀법

#? 단항식
res = np.polyfit(x, f(x), deg=7, full=True) #? 최적의 매개변수 결정
"""
#! np.polyfit() 매개변수
#* x : x 좌표 (독립 변수)
#* y : y 좌표 (종속 변수)
#* deg : 회귀 다항식의 차수
#* full : True면 추가적인 진단 정보 반환
#* w : y 좌표에 적용할 가중치
#* cov : True면 공분산행렬도 반환
"""
ry = np.polyval(res[0], x) #? 근찻값 계산

np.allclose(f(x), ry) #? 온벽히 일치하는지 확인 (False)

# create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'regression'], ['x', 'f(x)'])

#? 개별식
#! 목표 함수에 대해 어느정도 알고 있을 때 사용하여 학습 효과 극대화
matrix = np.zeros((4, len(x))) #? 기저 함수 정의
matrix[3, :] = np.sin(x) #* 목표함수와 비슷한 값을 부여
matrix[2, :] = x ** 2
matrix[1, :] = x
matrix[0, :] = 1

reg = np.linalg.lstsq(matrix.T, f(x), rcond=None)[0] #? 회귀 분석
ry = np.dot(reg, matrix) #? 함수에 대한 회귀분석 추정값

np.allclose(f(x), ry) #? 온벽히 일치하는지 확인 (True)

# create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'regression'], ['x', 'f(x)'])

# TODO: 보간법
#! x 차원의 정렬된 관측점이 주어졌을 때, 2개의 이웃하는 관측점 사이의 자료 계산하는 보간 함수를 찾는 과정
#! 각 관측점은 연속 미분 가능한 함수
#! 보간 함수는 큐빅 스플라인 함수

ipo = spi.splrep(x, f(x), k=1) #? 선형 스플라인 보간 구현
""" 
#! splrep() 매개변수
#* x : 정렬된 x 좌표(독립 변수)
#* y : 방향으로 정렬된 y 좌표(종속 변수)
#* w : y 좌효에 적용할 가중치
#* xb, xe : 적용 구간 (기본값은 [x[0],x[-1]])
#* k : 스플라인 함수의 차수 (1~5)
#* s : 숫자가 커질수록 부드러워짐
#* full_output : True면 추가적인 출력 반환
#* quiet : True면 메세지 미출력
"""
iy = spi.splev(x, ipo) #?  보간된 값 유도
""" 
#! splev() 매개변수
#* x : 정렬된 x 좌표(독립 변수)
#* tck : splerp가 반환한 값
#* der : 미분 차수
#* ext : x가 범위 밖인 경우의 행동 (0: 외삽, 1: 0으로 지정, 2: ValueError)
"""

np.allclose(f(x), iy) #? 온벽히 일치하는지 확인 (True)

create_plot([x, x], [f(x), iy], ['b', 'ro'], ['f(x)', 'interpolation'], ['x', 'f(x)'])
plt.show()

""" 
#? 보간법이 회귀법보다 더 정확한 근사가 가능
#? 하지만 데이터가 정렬되어 있어야 하고, 잡음이 없어야 한다.
#? 그리고 계산량이 많아 리소스가 많이 필요하다.
"""