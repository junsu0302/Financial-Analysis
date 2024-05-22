import math
import numpy as np
import numpy.random as npr
import scipy.stats as scs
import matplotlib.pyplot as plt

#! 몬테카를로 시뮬레이션은 수학식 or 파생상품의 가치를 계산할 때 가장 유연한 수치해석 방법론이다.

#? 두 데이터의 정보 비교
def print_statistics(a1, a2):
  sta1 = scs.describe(a1)
  sta2 = scs.describe(a2)
  print('%14s %14s %14s' % ('statistic', 'data set 1', 'data set 2'))
  print(45 * "-")
  print('%14s %14.3f %14.3f' % ('size', sta1[0], sta2[0])) #* 데이터셋 샘플 수
  print('%14s %14.3f %14.3f' % ('min', sta1[1][0], sta2[1][0])) #* 데이터셋 최솟값
  print('%14s %14.3f %14.3f' % ('max', sta1[1][1], sta2[1][1])) #* 데이터셋 최댓값
  print('%14s %14.3f %14.3f' % ('mean', sta1[2], sta2[2])) #* 데이터셋 평균
  print('%14s %14.3f %14.3f' % ('std', np.sqrt(sta1[3]), np.sqrt(sta2[3]))) #* 데이터셋 표준편차
  print('%14s %14.3f %14.3f' % ('skew', sta1[4], sta2[4])) #* 데이터셋 왜도 (분포의 비대칭 정도)
  print('%14s %14.3f %14.3f' % ('kurtosis', sta1[5], sta2[5])) #* 데이터셋 첨도 (분포의 집중 정도)

# TODO: 정적 시뮬레이션
#! 브라운 운동 모형 (확률 변수를 통한 미래 주가 시뮬레이션 - 확률 변수는 변경 x)
#! 주가의 확률 분포와 시간에 따른 변동성을 반영
#? 변수 초기화
S0 = 100 #* 최초 주가
r = 0.05 #* 무위험 단기 이자율
sigma = 0.25 #* 주가 변동성
T = 2.0 #* 시뮬레이션 기간 (2년)
I = 10000 #* 시뮬레이션 개수

#? 주가 시뮬레이션 (ST1: 정규분포기반, ST2: 로그정규분포 기반)
ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T +                    #* 주가의 평균 수익률 반영
                  sigma * math.sqrt(T) * npr.standard_normal(I))  #* 변동성 요소 반영
ST2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T,              #* 로그 정규 분포의 평균
                         sigma * math.sqrt(T),                    #* 로그 정규 분포의 표준편차
                         size=I)

# print_statistics(ST1, ST2)

# plt.figure(figsize=(10, 6))
# plt.hist(ST1, bins=50)
# plt.xlabel('index level')
# plt.ylabel('frequency')
# plt.show()

# TODO: 동적 시뮬레이션
#! 이산화: 정적 시뮬레이션 방정식에 미분방정식을 도입하여 아주 작은 시간간격으로 나누어 계산
#! 주가의 확률 분포와 변동성 분석

#? 변수 초기화
S0 = 100 #* 최초 주가
r = 0.05 #* 무위험 단기 이자율
sigma = 0.25 #* 주가 변동성

#? 시뮬레이션 변수 초기화
T = 2.0 #* 시뮬레이션 기간 (2년)
M = 50 #* 이산화 시간 구간의 개수
I = 10000 #* 시뮬레이션 개수
dt = T / M #* 각 시간 구간의 길이

#? 초깃값 설정
S = np.zeros((M + 1, I)) #* 주가를 저장할 객체
S[0] = S0 #* 최초 주가 설정

#? 주가 시뮬레이션 (브라운 운동 모형)
for t in range(1, M+1):
  S[t] = S[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt +                    #* 드리프트 항 (주가의 평균 수익률)
                         sigma * math.sqrt(dt) * npr.standard_normal(I))  #* 변동성 항 (주가의 무작위 변동성)
  
# plt.figure(figsize=(10, 6))
# plt.plot(S[:, :10], lw=1.5)
# plt.xlabel('time')
# plt.ylabel('index level')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.hist(S[-1], bins=50)
# plt.xlabel('index level')
# plt.ylabel('frequency')
# plt.show()

# print_statistics(S[-1], ST2)


# TODO: 제곱근 확산 모형
#! 단기 이자율, 변동성 모형에서 평균 회귀 과정이 사용된다. (제곱근 확산)
#! 단기 이자율 분포와 변동성 분석

#? 변수 초기화
x0 = 0.05 #* 단기 이자율 초깃값
kappa = 3.0 #* 평균 회귀계수 (theta로 회귀하는 속도 결정)
theta = 0.02 #* 장기 이자율 평균값 
sigma = 0.1 #* 이자율 변동성

#? 시뮬레이션 변수 초기화
T = 2.0 #* 시뮬레이션 기간 (2년)
M = 50 #* 이산화 시간 구간의 개수
I = 10000 #* 시뮬레이션 개수
dt = T / M #* 각 시간 구간의 길이

def srd_exact(): #? 이자율 시뮬레이션
  x = np.zeros((M+1, I))
  x[0] = x0
  for t in range(1, M+1):
    df = 4 * theta * kappa / sigma ** 2                         #* 자유도
    c = (sigma ** 2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)  #* 조정된 분산
    nc = np.exp(-kappa * dt) / c * x[t - 1]                     #* 비중신 모수
    x[t] = c * npr.noncentral_chisquare(df, nc, size=I)         #* 이자율 계산 (비중심 카이제곱 분포 기반)
  return x
  
result = srd_exact()

# plt.figure(figsize=(10, 6))
# plt.hist(result[-1], bins=50)
# plt.xlabel('value')
# plt.ylabel('frequency')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(result[:, :10], lw=1.5)
# plt.xlabel('time')
# plt.ylabel('index level')
# plt.show()

# TODO: 확률적 변동성 모형
#! 제곱근 확산 모형에 오일러 방식 사용
#! 레버리지 효과 추가 (시장이 하락할 때 변동성이 증가, 시장이 상승할 때 변동성이 감소)
#! 상관계수 행렬 + 숄레스키 분해 사용

#? 변수 초기화
S0 = 100. #* 초기 주가
r = 0.05 #* 무위험 이자율
v0 = 0.1 #* 초기 변동성
kappa = 3.0 #* 변동성 회귀 모형의 속도 파라미터
theta = 0.25 #* 장기 평균 변동성
sigma= 0.1 #* 변동성의 별도 요인
rho = 0.6 #* 주가와 변동성 사이의 상관계수

#? 시뮬레이션 변수 초기화
T = 1.0 #* 시뮬레이션 기간 (1년)
M = 50 #* 이산화 시간 구간의 개수
I = 10000 #* 시뮬레이션 개수
dt = T / M #* 각 시간 구간의 길이

#? 상관 관계 행렬 생성
corr_mat = np.zeros((2, 2)) #* 주가와 변동성 사이 상관 관계 반영
corr_mat[0, :] = [1.0, rho]
corr_mat[1, :] = [rho, 1.0]
cho_mat = np.linalg.cholesky(corr_mat) #* 상관 관계 행렬을 분해한 하삼각 행렬(숄레스키 분해 기반)

#? 3차원 난수 생성
ran_num = npr.standard_normal((2, M+1, I))

#? 초깃값 설정
v = np.zeros_like(ran_num[0]) #* 변동성 반영
vh = np.zeros_like(v) #* 변동성의 중간 값 반영
v[0] = v0
vh[0] = v0
S = np.zeros_like(ran_num[0])
S[0] = S0

#? 변동성 시뮬레이션
for t in range(1, M+1):
  ran = np.dot(cho_mat, ran_num[:, t, :]) #* 주가와 변동성의 상관된 무작위성 생성
  vh[t] = (vh[t - 1] + 
           kappa * (theta - np.maximum(vh[t-1], 0)) * dt +                    #* 변동성의 평균 회귀
           sigma * np.sqrt(np.maximum(vh[t-1], 0)) * math.sqrt(dt) * ran[1])  #* 변동성의 무작위 변동
  
v = np.maximum(vh, 0)

#? 주가 시뮬레이션
for t in range(1, M+1):
  ran = np.dot(cho_mat, ran_num[:, t, :]) #* 주가와 변동성의 상관된 무작위성 생성
  S[t] = S[t-1] * np.exp((r - 0.5 * v[t]) * dt +                #* 드리프트 항 (변동성으로 인한 평균 회귀)
                         np.sqrt(v[t]) * ran[0] * np.sqrt(dt))  #* 확산 항 (주가의 무작위 변동성)
  
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
# ax1.hist(S[-1], bins=50)
# ax1.set_xlabel('index level')
# ax1.set_ylabel('frequency')
# ax2.hist(v[-1], bins=50)
# ax2.set_xlabel('volatility')
# plt.show()

# print_statistics(S[-1], v[-1])

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
# ax1.plot(S[:, :10], lw=1.5)
# ax1.set_ylabel('index level')
# ax2.plot(v[:, :10], lw=1.5)
# ax2.set_xlabel('time')
# ax2.set_ylabel('volatility')
# plt.show()

# TODO: 점프 확산
#! 자산 가격이나 변동성이 크게 튀는 현상

#? 변수 초기화
S0 = 100. #* 초기 주가
r = 0.05 #* 연간 이자율
sigma = 0.2 #* 변동성
lamb = 0.75 #* 점프 강도 (단위 시간당 점프 발생률)
mu = -0.6 #* 평균 점프 크기
delta = 0.25 #* 점프 변동성
rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1) #* 표류계수 수정치 (점프의 위험 중립성 보존 -> 주가가 무작위로 변동하는 것 보정)

#? 시뮬레이션 변수 초기화
T = 1.0 #* 시뮬레이션 기간
M = 50 #* 시간 구간의 수
I = 10000 #* 시뮬레이션 경로 수
dt = T / M #* 각 시간 구간의 길이

#? 난수 생성 및 초깃값 설정
S = np.zeros((M+1, I))
S[0] = S0
sn1 = npr.standard_normal((M+1, I)) #* 표준정규분포 난수 (주가의 연속적인 변동 모형화)
sn2 = npr.standard_normal((M+1, I)) #* 표준정규분포 난수 (주가의 연속적인 변동 모형화)
poi = npr.poisson(lamb * dt, (M+1, I)) #* 포아송 분포 난수 (주가의 점프 모형화)

for t in range(1, M+1, 1):
  S[t] = S[t-1] * (np.exp((r - rj - 0.5 * sigma ** 2) * dt +    #* 표류항 (주가의 연속적 변동)
                          sigma * math.sqrt(dt) * sn1[t]) +     #* 확산항 (주가의 연속적 변동)
                          (np.exp(mu + delta * sn2[t]) - 1) *   #* 점프의 크기 (주가의 점프)
                          poi[t])                               #* 시간 구간 내 발생한 점프 횟수 (주가의 점프)
  S[t] = np.maximum(S[t], 0)

# plt.figure(figsize=(10, 6))
# plt.hist(S[-1], bins=50)
# plt.xlabel('value')
# plt.ylabel('frequency')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(S[:, :10], lw=1.5)
# plt.xlabel('time')
# plt.ylabel('index level')
# plt.show()

# TODO: 분산 감소
#! 모멘트 정합 : 난수의 경우 분산이 높아지므로 분산 감소를 위해 사용

def gen_sn(M, I, anti_paths=True, mo_match=True):
  """주어진 개수의 표준 정규 분포 난수 생성

  Args:
    M (int): 시간 구간의 개수
    I (int): 시뮬레이션 반복 횟수
    anti_paths (bool, optional): 반대 부호의 난수를 생성할지 여부 (기본값은 True)
    mo_match (bool, optional): 난수를 모멘트 매칭하여 평균을 0, 표준편차를 1로 반들지 여부 (기본값은 True)

  Returns:
    list: 생성된 표준 졍규 분포 난수 배열
  """
  if anti_paths is True:
    sn = npr.standard_normal((M+1, int(I / 2)))
    sn = np.concatenate((sn, -sn), axis=1)
  else:
    sn = npr.standard_normal((M+1, I))

  if mo_match is True:
    sn = (sn - sn.mean()) / sn.std()

  return sn

print('%15s %15s' % ('Mean', 'Std. Deviation'))
print(31 * "-")
for i in range(1, 31, 2):
  npr.seed(100)
  sn = gen_sn(50, 10000)
  print('%15.12f %15.12f' % (sn.mean(), sn.std()))