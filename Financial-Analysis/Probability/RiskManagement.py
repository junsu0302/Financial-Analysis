import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as scs

# TODO: VaR
#! 확률적인 신뢰도하에서 특정 시간 내에 발생할 수 있는 포트폴리오 or 단일 포지션의 손실 가능 금액

# TODO: 블랙-숄즈-머트 모형의 동적 시뮬레이션
#? 변수 초기화
S0 = 100 #* 최초 주가
r = 0.05 #* 무위험 단기 이자율
sigma = 0.25 #* 주가 변동성
T = 30 / 365. #* 시뮬레이션 기간 (2년)
M = 50 #* 시간 구간의 개수
I = 10000 #* 시뮬레이션 개수

#? 기하 브라운 운동의 만기 값 시뮬레이션
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T +
                 sigma * np.sqrt(T) * npr.standard_normal(I))

#? 시뮬레이션의 손익 계산 및 정렬
R_gbm = np.sort(ST - S0)

#? 시각화
# plt.figure(figsize=(10, 6))
# plt.hist(R_gbm, bins=50)
# plt.xlabel('absolution return')
# plt.ylabel('freauency')
# plt.show()

#? 신뢰 수준에 대한 최대 손실 출력
percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
var = scs.scoreatpercentile(R_gbm, percs)
# print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
# print(33 * '-')
# for pair in zip(percs, var):
#   print('%16.2f %16.3f' % (100-pair[0], -pair[1]))

# TODO: 점프 확산 모형의 동적 시뮬레이션
#? 변수 초기화
lamb = 0.75 #* 점프 강도 (단위 시간당 점프 발생률)
mu = -0.6 #* 평균 점프 크기
delta = 0.25 #* 점프 변동성
rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1) #* 표류계수 수정치 (점프의 위험 중립성 보존 -> 주가가 무작위로 변동하는 것 보정)
dt = 30. / 365 / M

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

#? 시뮬레이션의 손익 계산 및 정렬
R_jd = np.sort(S[-1] - S0)

#? 시각화
# plt.figure(figsize=(10, 6))
# plt.hist(R_jd, bins=50)
# plt.xlabel('absolute return')
# plt.ylabel('frequency')
# plt.show()

#? 신뢰 수준에 대한 최대 손실 출력
percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
var = scs.scoreatpercentile(R_jd, percs)
# print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
# print(33 * '-')
# for pair in zip(percs, var):
#   print('%16.2f %16.3f' % (100-pair[0], -pair[1]))

# TODO: 두 시뮬레이션의 신뢰 구간 시각화
percs = list(np.arange(0.0, 10.1, 0.1))
gbm_var = scs.scoreatpercentile(R_gbm, percs)
jd_var = scs.scoreatpercentile(R_jd, percs)

# plt.figure(figsize=(10, 6))
# plt.plot(percs, gbm_var, 'b', lw=1.5, label='GBM')
# plt.plot(percs, jd_var, 'r', lw=1.5, label='JD')
# plt.legend(loc=4)
# plt.xlabel('100 - confidence level [%]')
# plt.ylabel('value-at-risk')
# plt.ylim(ymax=0.0)
# plt.show()

# TODO: CVA (신용 VaR)
#! 거래 상대방이 이행의무를 다하지 않을 수 있는 가능성(부도)을 고려한 위험 측도

#? 변수 초기화
S0 = 100 #* 최초 주가
r = 0.05 #* 무위험 단기 이자율
sigma = 0.2 #* 주가 변동성
T = 1. #* 시뮬레이션 기간 (2년)
M = 50 #* 시간 구간의 개수
I = 100000 #* 시뮬레이션 개수

#? 기하 브라운 운동의 만기 값 시뮬레이션
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T +
                 sigma * np.sqrt(T) * npr.standard_normal(I))

#? 부도 변수 초기화
p = 0.01 #* 부도 확률
L = 0.5 #* 손실 수준
D = npr.poisson(p * T, I) #* 부도 사건 시뮬레이션
D = np.where(D > 1, 1, D) #* 값을 1로 제한

K = 100.
hT = np.maximum(ST - K, 0)

#? CVA 시뮬레이션
C0 = math.exp(-r * T) * np.mean(hT) #* 유러피안 콜 옵션에 대한 몬테카를로 추정 가치
CVaR = math.exp(-r * T) * np.mean(L * D * hT) #* 부도 사건이 있을 때 미래 손실의 할인 평균액
S0_CVA = math.exp(-r * T) * np.mean((1 - L * D) * hT) #* 유러피안 콜 옵션에 대한 몬테카를로 추정 가치(부도 사건으로 인한 손실 조정)

#? 부도 데이터 시각화
np.count_nonzero(D) #* 부도의 수
np.count_nonzero(L * D * hT) #* 부도로 인한 손실
I - np.count_nonzero(hT) #* 부도로 옵션 가치가 없어지는 경우

plt.figure(figsize=(10, 6))
plt.hist(L * D * hT, bins=50)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.ylim(ymax=350)

plt.show()