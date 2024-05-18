import math
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

#TODO: 최적화
#! 최적화에서 전역 최소화를 진행한 후에 국소 최적화를 진행한다. (하나의 국소 최소점에 빠져 다른 값을 탐색하지 못하기 때문)
def fm(p): #? 목표 함수
  x, y = p
  return (np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2)

def fo(p): #? 목표 함수 + 연산 정보 출력
  x, y = p
  z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
  print('%8.4f | %8.4f | %8.4f' % (x, y, z))
  return z

# TODO: 전역 최적화
# opt1 = sco.brute(fo, ((-10, 10.1, 0.1), (-10, 10.1, 0.1)), finish=None) #? 최적화를 통해 구한 최적 인숫값 : [-1.4, -1.4]
# fm(opt1) #? 최소화 지점 함수값 : -1,7749

# TODO: 국소 최적화
# opt2 = sco.fmin(fo, opt1, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20)
# print(opt2)
# fm(opt2)

# TODO: 주식 투자의 기대 효용함수 최적화
""" 
#? 상황 가정
#* 두 주식(q)은 모두 10의 비용을 요구한다.
#* 1년후 주식들은 다음 상태에서 해당 가치(r)를 가진다.
#*    상태 u: (15, 5)
#*    상태 d: (5, 12)
#* 투자자의 예산(w)는 100이다.
"""

def Eu(p): #? 기대 효용을 최대화하기 위한 최적화 함수
  s, b = p #? 현재 보유 수
  return -(0.5 * math.sqrt(s * 15 + b * 5) + 0.5 * math.sqrt(s * 5 + b * 12)) #? sqrt를 통해 최소화를 기반으로 최소화

cons = ({'type': 'ineq', 'fun': lambda p: 100 - p[0] * 10 - p[1] * 10}) #? 제한 조건
"""
#* 'type': 'ineq' : 제약 조건이 부등식임을 나타냄
#* 'fun': lambda p: 100 - p[0] * 10 - p[1] * 10 : 구매 비용이 100을 초과하지 않음
"""

bnds = ((0, 1000), (0, 1000)) #? 경계 조건 인수
#* 각 인수는 0 ~ 1000의 값을 갖을 수 있음

result = sco.minimize(Eu, [5, 5], method='SLSQP', bounds=bnds, constraints=cons) #? 최적화
"""
#? mininize()
#* fun : 최적화할 대상
#* x0 : 초기 추정값
#* method : 최적화 알고리즘 정의 (SLSQP : 경계와 제약 조건 처리)
#* bounds : 변수의 경계 지정
#* constraints : 제악 조건 정의
"""

"""
#? minimize()의 결과
#* fun : 최적화된 기대 효용 함수 값 (최소화를 진행 했으므로 해당 값의 절대값)
#* x : 최적화된 변수 값
#* nit : 반복 횟수
#* jac : 목적 함수의 최종 기울기 벡터
#* nfev : 목적 함수의 평가 횟수
#* njev : 기울기 평가 횟수
"""

result['x'] #? [8.025, 1.975] : 1번 주식 8개, 2번 주식 2개가 최적의 값임
-result['fun'] #? 9.700 : 기대 효용 값이 9.7임