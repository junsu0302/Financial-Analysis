import math
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.optimize as sco
import scipy.interpolate as sci
import matplotlib.pyplot as plt

raw = pd.read_csv('data/tr_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()
symbols = ['AAPL.O', 'MSFT.O', 'SPY', 'GLD']
"""
#* AAPL.O : 애플 주가
#* MSFT.O : 마이크로소프트 주가
#* SPY : SPDR S&P 500 ETF 주가
#* GLD : SPDR Gold ETF 주가
"""
noa = len(symbols)
data = raw[symbols]
rets = np.log(data / data.shift(1))

# TODO: 기초 포트폴리오

#? 포트폴리오 비중 설정
weights = npr.random(noa) #* 무작위 포트폴리오 비중 설정
weights /= np.sum(weights) #* 1로 정규화

#? 포트폴리오 연산
np.sum(rets.mean() * weights) * 252 #* 연율화된 포트폴리오 수익
np.dot(weights.T, np.dot(rets.cov() * 252, weights)) #* 연률화된 포트폴리오 분산
math.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))) #* 연율화된 포트폴리오 변동

#? 포트폴리오 수익률 계산
def port_ret(weights):
  return np.sum(rets.mean() * weights) * 252

#? 포트폴리오 변동성 계산
def port_vol(weights):
  return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

#? 무작위 포트폴리오의 수익률 및 변동성 계산
prets = []
pvols = []
for p in range(2500):
  weights = npr.random(noa)
  weights /= np.sum(weights)
  prets.append(port_ret(weights))
  pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)

#? 무작위 포트폴리오 비중에 대한 수익률과 변동성의 기댓값 시각화
# plt.figure(figsize=(10, 6))
# plt.scatter(pvols, prets, c=prets / pvols, marker='o', cmap='coolwarm')
# plt.xlabel('expected volatility')
# plt.ylabel('expected return')
# plt.colorbar(label='Sharpe ratio')
# plt.show()

# TODO: 포트폴리오 최적화

#? Sharpe 비율 최대화
#! Sharpe는 기대 수익률 / 변동성 이므로, 수익률 대비 위험을 최소화
def min_func_sharpe(weights):
  return -port_ret(weights) / port_vol(weights)

#? 제약 조건 정의
cons = ({'type': 'eq', 'fun': lambda x:np.sum(x) - 1}) #* 가중치의 합이 1 (모든 포트폴리오)

#? 경계 설정
bnds = tuple((0, 1) for x in range(noa)) 

#? 초기 가중치 설정
eweights = np.array(noa * [1. / noa,])

#? 포트폴리오 최적화 (Sharpe 비율 최대화)
opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons) #* 포트폴리오 결과
opts['x'] #* 최적 포트폴리오 비중
port_ret(opts['x']) #* 최적 포트폴리오 수익률
port_vol(opts['x']) #* 최적 포트폴리오 변동성
port_ret(opts['x']) / port_vol(opts['x']) #* 최대 샤프 지수

#? 포트폴리오 최적화 (변동성 최소화)
optv = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
optv['x'] #* 최적 포트폴리오 비중
port_ret(optv['x']) #* 최적 포트폴리오 수익률
port_vol(optv['x']) #* 최적 포트폴리오 변동성
port_ret(optv['x']) / port_vol(opts['x']) #* 최대 샤프 지수

# TODO: 효율적 투자선
#! 모든 최소 분산 포트폴리오보다 수익률이 높은 모든 최적 포트폴리오

#? 제약 조건 정의
cons = ({'type': 'eq', 'fun': lambda x:port_ret(x) - tret},
        {'type': 'eq', 'fun': lambda x:np.sum(x) - 1})

#? 최적의 가중치 계산
trets = np.linspace(0.05, 0.2, 50) #* 목표 수익률의 범위 정의
tvols = [] #* 최적 포트폴리오 변동성 리스트
#* 각 수익률에 대한 포트폴리오의 변동성을 최소화
for tret in trets:
  res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
  tvols.append(res['fun'])

#? 시각화
# plt.figure(figsize=(10, 6))
# plt.scatter(pvols, prets, c=prets / pvols, marker='.', alpha=0.8, cmap='coolwarm') #* Sharpe 비율
# plt.plot(tvols, trets, 'b', lw=4.0) #* 목표 수익률에 대한 최소 변동성을 갖는 포트폴리오
# plt.plot(port_vol(opts['x']), port_ret(opts['x']), 'y*', markersize=15.0) #* Sharpe 비율이 최대인 포트폴리오
# plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize=15.0) #* 변동성이 최소인 포트폴리오
# plt.xlabel('expected volatility')
# plt.ylabel('expected return')
# plt.colorbar(label='Sharpe ratio')
# plt.show()

# TODO: 자본시장선

ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]

tck = sci.splrep(evols, erets)

def f(x):
  """효율적 투자선 함수 (스플라인 근사))"""
  return sci.splev(x, tck, der=0)

def df(x):
  """효율적 투자선 1차 도함수"""
  return sci.splev(x, tck, der=1)

def equations(p, rf=0.01):
  eq1 = rf - p[0]
  eq2 = rf + p[1] * p[2] - f(p[2])
  eq3 = p[1] - df(p[2])
  return eq1, eq2, eq3

opt = sco.fsolve(equations, [0.01, 0.5, 0.15])

# plt.figure(figsize=(10, 6))
# plt.scatter(pvols, prets, c=(prets - 0.01) / pvols, marker='.', cmap='coolwarm')
# plt.plot(evols, erets, 'b', lw=4.0)
# cx = np.linspace(0.0, 0.3)
# plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5)
# plt.plot(opt[2], f(opt[2]), 'y*', markersize=15.0)
# plt.grid(True)
# plt.axhline(0, color='k', ls='--', lw=2.0)
# plt.axvline(0, color='k', ls='--', lw=2.0)
# plt.xlabel('expected volatility')
# plt.ylabel('expected return')
# plt.colorbar(label='Sharpe ratio')
# plt.show()

# TODO: 최종 포트폴리오 최적화

#? 데이터 로드 및 전처리
raw = pd.read_csv('data/tr_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna() #* CSV 파일 로드
symbols = ['AAPL.O', 'MSFT.O', 'SPY', 'GLD'] #* 사용할 주식 정의
data = raw[symbols] #* 데이터 추출
noa = len(symbols)
rets = np.log(data / data.shift(1)).dropna() #* 로그 수익률 계산

#? 포트폴리오 수익률과 변동성 계산 함수
def port_ret(weights):
  """주어진 가중치에서 포트폴리오의 연율화된 수익률 계산"""
  return np.sum(rets.mean() * weights) * 252

def port_vol(weights):
  """주어진 가중치에서 포트폴리오의 연율화된 변동성 계산"""
  return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

#? Sharpe 비율 최대화 함수
def min_func_sharpe(weights):
  """포트폴리오의 Sharpe 비율을 최대화하기 위한 목표함수"""
  return -port_ret(weights) / port_vol(weights)

#? 제약 조건과 경계 설정
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #* 제약조건 : 모든 가중치의 합이 1
bnds = tuple((0, 1) for _ in range(noa)) #* 각 가중치는 0~1
eweights = np.array(noa * [1. / noa]) #* 초기 가중치 설정

#? 포트폴리오 최적화 (Sharpe 비율 최대화)
opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)

#? 포트폴리오 최적화 (변동성 최소화)
optv = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)

#? 무작위 포트폴리오 생성 및 시각화
prets, pvols = [], []
for _ in range(2500):
  weights = npr.random(noa) #* 무작위 포트폴리오 비중 설정
  weights /= np.sum(weights) #* 비중의 합이 1이 되도록 정규화
  prets.append(port_ret(weights)) #* 포트폴리오 수익률 계산
  pvols.append(port_vol(weights)) #* 포트폴리오 변동성 계산

prets = np.array(prets)
pvols = np.array(pvols)

#? 효율적 투자선 계산
trets = np.linspace(0.05, 0.2, 50) #* 목표 수익률 범위 설정
tvols = []
for tret in trets:
  cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
  res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
  tvols.append(res['fun'])

#? 자본시장선 변수 정의
ind = np.argmin(tvols) #* 최소 변동성을 갖는 인덱스 탐색
evols = tvols[ind:] #* 효율적 변동성
erets = trets[ind:] #* 효율적 수익률

#? 효율적 투자선 스플라인 근사
tck = sci.splrep(evols, erets)

def f(x):
  """스플라인 근사를 통한 효울적 투자선 함수"""
  return sci.splev(x, tck, der=0)

def df(x):
  """효율적 투자선의 1차 도함수"""
  return sci.splev(x, tck, der=1)

def equations(p, rf=0.01):
  """자본시장선 계싼을 위한 방정식 정의"""
  eq1 = rf - p[0] #* 무위험 수익률 방정식
  eq2 = rf + p[1] * p[2] - f(p[2]) #* 자본시장선 방정식
  eq3 = p[1] - df(p[2]) #* 자본시장선 기울기 방정식
  return eq1, eq2, eq3

#? 자본시장선 풀이
opt = sco.fsolve(equations, [0.01, 0.5, 0.15])

#? 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=(prets - 0.01) / pvols, marker='.', cmap='coolwarm') #* 무작위 포트폴리오
plt.plot(evols, erets, 'b', lw=4.0) #* 효율적 투자선
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5) #* 자본시장선
plt.plot(port_vol(opts['x']), port_ret(opts['x']), 'y*', markersize=15.0) #* Sharpe 비율 최대 포트폴리오
plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize=15.0) #* 최소 변동성 포트폴리오
plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()
