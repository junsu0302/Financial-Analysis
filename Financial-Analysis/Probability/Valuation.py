import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# TODO: 가치 평가
#! 금융 옵션은 지정된 금융 상품을 만기일(유러피안 옵션) or 특정 기간(아메리칸 옵션) 동안 
#! 주어진 가격에 사거나(콜 옵션) 팔(풋 옵션) 수 있는 권리를 의미한다.
#! 일반적으로 아메리칸 옵션이 더 복잡하지만 유연성과 가치가 높다.
"""
#? 기하 브라운 운동 : 주식 가격[S(t)]의 동작을 모형화
#? 몬테카를로 시뮬레이션 : 여러 경로를 통해 예측한 미래 주가와 각 경로에서의 페이오프의 평균
#? 블랙-숄즈 모델 : 유러피안 옵션의 가격을 닫힌 형태로 계산
"""

def gen_sn(M:int, I:int, anti_paths:bool=True, mo_match:bool=True) -> list: #? 분산 감소
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


# TODO: 유러피안 옵션
#? 변수 초기화
S0 = 100. #* 초기 주가
r = 0.05 #* 무위험 이자율
sigma = 0.25 #* 변동성
T = 1.0 #* 옵션 만기 기간
M = 50 #* 시간 구간의 개수
I = 50000 #* 시뮬레이션 반복 횟수

def gbm_mcs_stat(K:float) -> float:
  """몬테카를로 시뮬레이션에 의한 유러피안 콜 옵션의 가치 평가

  Args:
    K (float): (양의) 옵션 행사가

  Returns:
    float: 유러피안 콜 옵션의 현재 추정 가치
  """
  sn = gen_sn(1, I)
  #? 주식 가격 시뮬레이션 (기하 브라운 운동)
  ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T
                   + sigma * math.sqrt(T) * sn[1])
  #? 만기에서의 유러피안 페이오프 계산
  hT = np.maximum(ST - K, 0)
  #? 몬테카를로 시뮬레이션 계산
  C0 = math.exp(-r * T) * np.mean(hT)
  return C0

def gbm_mcs_dyna(K:float, option:str) -> float:
  """몬테카를로 시뮬레이션에 의한 유러피안 옵션의 가치 평가

  Args:
    K (float): (양의) 옵션 행사가
    option (str, optional): 가치를 쳥가할 옵션의 유형 ('call', 'put')
      
  Returns:
    float: 옵션의 현재 가치 추정치
  """
  dt = T / M
  #? 지수 수준의 시뮬레이션
  S = np.zeros((M+1, I))
  S[0] = S0
  sn = gen_sn(M, I)
  for t in range(1, M+1):
    S[t] = S[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt
                           + sigma * math.sqrt(dt) * sn[t])
    
  #? 옵션 유형에 따른 페이오프 계산
  if option == 'call':
    hT = np.maximum(S[-1] - K, 0)
  else:
    hT = np.maximum(K - S[-1], 0)

  #? 몬테카를로 시뮬레이션 계산
  C0 = math.exp(-r * T) * np.mean(hT)
  return C0

def bms_call_value(S0:float, K:float, T:float, r:float, sigma:float) -> float:
  """BMS 모형에 의한 유러피안 콜 옵션의 가치 평가

  Args:
    S0 (float): 초기 주가/지수 수준
    K (float): 행사가
    T (float): 만기
    r (float): 고정 단기 무위험 이자율
    sigma (float): 변동성

  Returns:
    float: 유러피안 콜 옵션의 현재 가치
  """
  from math import log, sqrt, exp
  from scipy import stats
  
  #? 블랙-숄즈 모델을 통한 유러피안 콜 옵션의 가치 계산
  S0 = float(S0)
  d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T)) #* 욥션 가격의 불확실성 (시간, 변동성 반영)
  d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T)) #* 옵션의 실제 행사 확률 조정

  result = (S0 * stats.norm.cdf(d1, 0.0, 1.0) -
            K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
  return result

#? 리스트 초기화
stat_res = [] #* 정적 몬테카를로 시뮬레이션 결과 저장
dyna_res = [] #* 동적 몬테카를로 시뮬레이션 결과 저장
anal_res = [] #* 블랙-숄즈 모델을 사용한 분석 결과 저장
k_list = np.arange(80., 120.1, 5.) #* 행사 가격 리스트
npr.seed(100)

#? 행사가에 대한 옵션 가치 계산
for K in k_list:
  stat_res.append(gbm_mcs_stat(K))
  dyna_res.append(gbm_mcs_dyna(K, option='call'))
  anal_res.append(bms_call_value(S0, K, T, r, sigma))

stat_res = np.array(stat_res)
dyna_res = np.array(dyna_res)
anal_res = np.array(anal_res)

#? 결과 시각화 (정적 유러피안 분석)
# fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(12, 8))
# ax1.plot(k_list, anal_res, 'b', label='analytical')
# ax1.plot(k_list, stat_res, 'ro', label='static')
# ax1.set_ylabel('European call option value')
# ax1.legend(loc=0)
# ax1.set_ylim(bottom=0)
# wi = 1.0
# ax2.bar(k_list - wi / 2, (anal_res - stat_res) / anal_res * 100, wi)
# ax2.set_xlabel('strike')
# ax2.set_ylabel('difference in %')
# ax2.set_xlim(left=75, right=125)

#? 결과 시각화 (동적 유러피안 분석)
# ax3.plot(k_list, anal_res, 'b', label='analytical')
# ax3.plot(k_list, dyna_res, 'ro', label='dynamic')
# ax3.set_ylabel('European call option value')
# ax3.legend(loc=0)
# ax3.set_ylim(bottom=0)
# wi = 1.0
# ax4.bar(k_list - wi / 2, (anal_res - dyna_res) / anal_res * 100, wi)
# ax4.set_xlabel('strike')
# ax4.set_ylabel('difference in %')
# ax4.set_xlim(left=75, right=125)

# plt.show()

#TODO: 아메리칸 옵션
def gbm_mcs_amer(K:float, option:str) -> float:
  """LMS 알고리즘을 이용한 몬테카를로 시뮬레이션을 이용한 아메리칸 옵션의 가치 평	

  Args:
      K (float): (양의) 옵션 행사가
      option (str): 가치 평가할 옵션의 유형 ('call', 'put')
  
  Returns:
    float: 옵션의 현재 가치 추정
  """
  #? 파라미터 설정
  dt = T / M #* 각 시간 구간의 길이
  df = math.exp(-r * dt) #* 한 구간 당 할인율

  #? 지수 수준의 시뮬레이션
  S = np.zeros((M+1, I)) #* 각 시간 구간에서의 주가 경로
  S[0] = S0
  sn = gen_sn(M, I)
  for t in range(1, M+1):
    S[t] = S[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt
                           + sigma * math.sqrt(dt) * sn[t])
    
  #? 옵션 유형에 따른 페이오프 계산
  if option == 'call':
    h = np.maximum(S - K, 0)
  else:
    h = np.maximum(K - S, 0)
  
  #? LMS 알고리즘
  V = np.copy(h) #? 각 시간 구간에서의 옵션 가치
  for t in range(M-1, 0, -1):
    reg = np.polyfit(S[t], V[t+1] * df, 7) #? 다항 회귀 값 계산
    C = np.polyval(reg, S[t]) #? 회귀 값 계산
    V[t] = np.where(C > h[t], V[t+1] * df, h[t]) 
  
  #? 몬테카를로 시뮬레이션에 의한 추정치
  C0 = df * np.mean(V[1])
  return C0

euro_res = []
amer_res = []
k_list = np.arange(80., 120.1, 5.)

for K in k_list:
  euro_res.append(gbm_mcs_dyna(K, 'put'))
  amer_res.append(gbm_mcs_amer(K, 'put'))

euro_res = np.array(euro_res)
amer_res = np.array(amer_res)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1.plot(k_list, euro_res, 'b', label='European put')
ax1.plot(k_list, amer_res, 'ro', label='American put')
ax1.set_ylabel('call option value')
ax1.legend(loc=0)
wi = 1.0
ax2.bar(k_list - wi / 2, (amer_res - euro_res) / euro_res * 100, wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('early exercise premium in %')
ax2.set_xlim(left=75, right=125)
plt.show()