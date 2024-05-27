import math
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pyplot as plt
""" 
# TODO: 정규성 검사
#! 다음과 같은 금융 이론들은 모두 주식 시장의 수익률이 정규분포를 이룬다는 사실에 기반하고 있다.
#? 포토폴리오 이론
#? 자본 자산 가격결정 모형
#? 효율적 시장 가설
#? 옵션 가격결정 모형
"""

# TODO: 벤치마크 자료 분석

def gen_paths(S0:float, r:float, sigma:float, T:float, M:int, I:int) -> np.ndarray:
  """기하 브라운 운동에 대한 몬테카를로 경로 생성

  Args:
    S0 (float): 초기 주가
    r (float): 고정 단기 이자율
    sigma (float): 고정 변동성
    T (float): 최종 시간
    M (int): 시간 구간의 개수
    I (int): 시뮬레이션 경로의 개수

  Returns:
    np.ndarray: 주어진 인수로 시뮬레이션한 경로
  """
  #? 변수 정의
  dt = T / M #* 각 구간의 길이
  paths = np.zeros((M+1, I)) #* 경로 배열
  paths[0] = S0 #* 초기 주가 설정

  #? 시뮬레이션
  for t in range(1, M+1):
    rand = npr.standard_normal(I) #* I개의 표준정규분포 난수 생성
    rand = (rand - rand.mean()) / rand.std() #* 정규화하여 평균 0, 표준편차 1로 설정
    #? 기하 브라운 운동
    paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + #* 드리프트 항 : 평균 수익률 반영
                                   sigma * math.sqrt(dt) * rand) #* 확산 항 : 주가의 변동성 반영
  return paths

def print_statistics(arr:np.ndarray):
  """선택한 통계를 출력

  Args:
    array (np.ndarray): 통계를 생성할 대상 객체
  """
  sta = scs.describe(arr)
  print('%14s %15s' % ('statistic', 'value'))
  print(30 * '-')
  print('%14s %15.5f' % ('size', sta[0]))
  print('%14s %15.5f' % ('min', sta[1][0]))
  print('%14s %15.5f' % ('max', sta[1][1]))
  print('%14s %15.5f' % ('mean', sta[2]))
  print('%14s %15.5f' % ('std', np.sqrt(sta[3])))
  print('%14s %15.5f' % ('skew', sta[4]))
  print('%14s %15.5f' % ('kurtosis', sta[5]))

def normality_tests(arr:np.ndarray):
  """주어진 데이터 분포의 정규성 검정

  Args:
    arr (np.ndarray): 통계를 생성할 대상 객체
  """
  arr = arr.flatten()
  print('%20s %14.3f' % ('Skew of data set', scs.skew(arr)))
  _, skew_p_value = scs.skewtest(arr) #? 왜도 검정
  print('%20s %14.3f' % ('Skew test p-value', skew_p_value)) 
  print('%20s %14.3f' % ('Kurt of data set', scs.kurtosis(arr)))
  _, kurt_p_value = scs.kurtosistest(arr) #? 첨도 검정
  print('%20s %14.3f' % ('Kurt test p-value', kurt_p_value)) 
  _, norm_p_value = scs.normaltest(arr) #? 정규성 검정
  print('%20s %14.3f' % ('Norm test p-value', norm_p_value))

#? 초기 파라미터 설정
S0 = 100. #* 초기 주가
r = 0.05 #* 고정 단기 이자율
sigma = 0.2 #* 고정 변동성
T = 1.0 #* 최종 시간
M = 50 #* 시간 구간의 개수
I = 250000 #* 시뮬레이션 경로의 수
np.random.seed(1000)

#? 경로 생성
paths = gen_paths(S0, r, sigma, T, M, I)

#? 데이터 분석
log_data = np.log(paths[-1]) #* 주가의 절대값 분석
log_returns = np.log(paths[1:] / paths[:-1]) #* 주가의 로그 수익률(변화율) 분석

#? 통계값 확인
# print_statistics(log_data)
# print_statistics(log_returns.flatten())

#? 정규성 검정
# print('Normality Tests')
# normality_tests(log_data)
# normality_tests(log_returns.flatten())

#? 시뮬레이션 경로 시각화
# plt.figure(figsize=(10, 6))
# plt.plot(paths[:, :10])
# plt.xlabel('time steps')
# plt.ylabel('index level')
# plt.show()

# TODO: 데이터 기반 정규화 검정

# TODO: 데이터 준비
#? 데이터 준비
raw = pd.read_csv('data/tr_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()
symbols = ['SPY', 'GLD', 'AAPL.O', 'MSFT.O']
"""
#* SPY : SPDR S&P 500 ETF 주가
#* GLD : SPDR Gold ETF 주가
#* AAPL.O : 애플 주가
#* MSFT.O : 마이크로소프트 주가
"""
data = raw[symbols]
data = data.dropna()

#? 금융 상품 가격 데이터 시각화
# (data / data.iloc[0] * 100).plot(figsize=(10, 6)) #* 초기값을 100으로 정규화
# plt.xlabel('Date') #* x축 레이블 추가
# plt.ylabel('Normalized Price') #* y축 레이블 추가
# plt.title('Normalized Prices of Selected Stocks') #* 제목 추가
# plt.legend(symbols) #* 범례 추가
# plt.show()

#? 금융 상품의 수익률 시각화
log_returns = np.log(data / data.shift(1))
# log_returns.hist(bins=50, figsize=(10, 6))
# plt.show()

#? 금융 상품 시계열의 통계치 시각화
# for sym in symbols:
#   print('\nResults for symbol {}'.format(sym))
#   print('-' * 30)
#   log_data = np.array(log_returns[sym].dropna())
#   print_statistics(log_data)

#? 금융 상품 시계열의 정규성 검정 시각화
for sym in symbols:
  print('\nResults for symbol {}'.format(sym))
  print('-' * 35)
  log_data = np.array(log_returns[sym].dropna())
  normality_tests(log_data)