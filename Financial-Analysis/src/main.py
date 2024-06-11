import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as spo

from Environment.market_environment import MarketEnvironment
from Utils.constant_short_rate import ConstantShortRate
from Simulation.jump_diffusion import JumpDiffusion
from Valuation.mcs_european import MCSEuropean

from Utils.derivatives_position import DerivativesPosition
from Utils.derivatives_portfolio import DerivativesPortfolio

# TODO: 옵션 데이터

#? 데이터 읽기
dax = pd.read_csv('data/tr_eikon_option_data.csv')

#? 날짜 컬럼 형식 변환
for col in ['CF_DATE', 'EXPIR_DATE']:
  dax[col] = dax[col].apply(lambda date: pd.Timestamp(date))

#? 초기 가격 설정
initial_value = dax.iloc[0]['CF_CLOSE']

#? Call과 Put 옵션 분리
calls = dax[dax['PUTCALLIND'].str.strip() == 'CALL'].copy()
puts = dax[dax['PUTCALLIND'].str.strip() == 'PUT'].copy()

"""
#? 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1 = calls.set_index('STRIKE_PRC')[['CF_CLOSE', 'IMP_VOLT']].plot(secondary_y='IMP_VOLT', style=['bo', 'rv'], ax=ax1)
ax1.set_title('Call Options')
ax1.set_xlabel('Strike Price')
ax1.set_ylabel('Close Price')
ax1.right_ax.set_ylabel('Implied Volatility')

ax2 = puts.set_index('STRIKE_PRC')[['CF_CLOSE', 'IMP_VOLT']].plot(secondary_y='IMP_VOLT', style=['bo', 'rv'], ax=ax2)
ax2.set_title('Put Options')
ax2.set_xlabel('Strike Price')
ax2.set_ylabel('Close Price')
ax2.right_ax.set_ylabel('Implied Volatility')

plt.tight_layout()
plt.show()
"""

# TODO: 모형 보정

#? 시장 데이터 선택
limit = 500
option_selection = calls[abs(calls['STRIKE_PRC'] - initial_value) < limit].copy()

#? 옵션 모델 설정
pricing_date = option_selection['CF_DATE'].max()
me_dax = MarketEnvironment('DAX30', pricing_date)

#? Jump Diffusion 모델 파라미터 설정
maturity = pd.Timestamp(calls.iloc[0]['EXPIR_DATE']) #* 만기일
me_dax.add_constant('initial_value', initial_value) #* 초기 값 설정
me_dax.add_constant('final_date', maturity) #* 만기일 설정
me_dax.add_constant('currency', 'EUR') #* 통화 설정
me_dax.add_constant('frequency', 'B') #* 주기 설정
me_dax.add_constant('paths', 10000) #* 경로 수 설정
csr = ConstantShortRate('csr', 0.01) #* 1% 이자율 할인곡선 생성
me_dax.add_curve('discount_curve', csr) #* 할인곡선 설정
me_dax.add_constant('volatility', 0.2) #* 변동성 설정
me_dax.add_constant('lambda', 0.8) #* 람다 설정
me_dax.add_constant('mu', -0.2) #* 뮤 설정
me_dax.add_constant('delta', 0.1) #* 델타 설정
dax_model = JumpDiffusion('dax_model', me_dax) #* 모델 생성
me_dax.add_constant('strike', initial_value) #* 행사가 설정
me_dax.add_constant('maturity', maturity) #* 만기일 설정
payoff_func = 'np.maximum(maturity_value - strike, 0)' #* 옵션 페이오프 설정
dax_eur_call = MCSEuropean('dax_eur_call', dax_model, me_dax, payoff_func) #* 유러피안 콜 옵션 생성
option_models = {}  #* 각 행사가에 대한 옵션 모델 생성
for option in option_selection.index:
  strike = option_selection['STRIKE_PRC'].loc[option]
  me_dax.add_constant('strike', strike)
  option_models[strike] = MCSEuropean('eur_call_%d' % strike, dax_model, me_dax, payoff_func)

#? 모형 보정
def calculate_model_values(p0:tuple) -> dict:
  """주어진 파라미터로 모형을 업데이트하고 각 행사 가격에 대한 옵션 가치를 계산하여 반환

  Args:
    p0 (tuple): 모형 파라미터 (volatility, lamb, mu, delta)

  Returns:
    dict: 각 행사 가격에 대한 모형으로 계산된 옵션 가치
  """
  volatility, lamb, mu, delta = p0
  dax_model.update(volatility=volatility, lamb=lamb, mu=mu, delta=delta)
  return {
    strike: model.present_value(fixed_seed=True)
    for strike, model in option_models.items()
  }

#? 보정 절차 및 최적화
i = 0
def mean_squared_error(p0:tuple) -> float:
  """
  주어진 모형 파라미터에 대한 평균 제곱 오차 계산.
  해당 함수는 최적화 알고리즘이 최소화하려는 목적함수이다.

  Args:
    p0 (tuple): 모형 파라미터 (volatility, lamb, mu, delta)

  Returns:
    float: 모형에 대한 평균 제곱 오차
  """
  global i
  model_values = np.array(list(calculate_model_values(p0).values()))
  market_values = option_selection['CF_CLOSE'].values
  option_diffs = model_values - market_values
  
  MSE = np.sum(option_diffs ** 2) / len(option_diffs)

#   if i % 75 == 0:
#     if i == 0:
#       print('%4s %6s %6s %6s %6s --> %6s' % ('i', 'vola', 'lambda', 'mu', 'delta', 'MSE'))
#     print('%4d %6.3f %6.3f %6.3f %6.3f --> %6.3f' % (i, p0[0], p0[1], p0[2], p0[3], MSE))

  i += 1
  return MSE

#? 최적화
i = 0
#* 전역 최적화
opt_global = spo.brute(mean_squared_error,      #* 목표 함수
                       ((0.10, 0.201, 0.025),   #* volatility 범위
                        (0.10, 0.80, 0.10),     #* lambda 범위
                        (-0.40, 0.01, 0.10),    #* mu 범위
                        (0.00, 0.121, 0.02)),   #* delta 범위
                        finish=None)
#* 지역 최적화
opt_local = spo.fmin(mean_squared_error,        #* 목표 함수
                     opt_global,                #* 시작 지점
                     xtol=0.00001,              #* 기울기 값에 대한 허용 오차
                     ftol=0.00001,              #* 기준 함수 값에 대한 허용 오차
                     maxiter=200,               #* 최대 반복 횟수
                     maxfun=550)                #* 최대 함수 호출 횟수

#? 시각화를 포함한 전처리
option_selection['MODEL'] = np.array(list(calculate_model_values(opt_local).values()))
option_selection['ERRORS_EUR'] = (option_selection['MODEL'] - option_selection['CF_CLOSE'])
option_selection['ERRORS_%'] = (option_selection['ERRORS_EUR'] / option_selection['CF_CLOSE']) * 100

"""
#? 시각화 설정
fix, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 10))
strikes = option_selection['STRIKE_PRC'].values
wi = 15

#? 첫 번째 그래프: 시장 가격 및 모델 값
ax1.plot(strikes, option_selection['CF_CLOSE'], label='Market Quotes', color='blue')
ax1.plot(strikes, option_selection['MODEL'], 'ro', label='Model Values', markersize=5)
ax1.set_ylabel('Option Values', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.set_title('Market Quotes vs. Model Values', fontsize=14)

#? 두 번째 그래프: 오차 (유로 단위)
ax2.bar(strikes - wi / 2., option_selection['ERRORS_EUR'], width=wi, color='orange')
ax2.set_ylabel('Errors [EUR]', fontsize=12)

#? 세 번째 그래프: 오차 (백분율)
ax3.bar(strikes - wi / 2., option_selection['ERRORS_%'], width=wi, color='green')
ax3.set_ylabel('Errors [%]', fontsize=12)
ax3.set_xlabel('Strikes', fontsize=12)

#? 시각화
plt.tight_layout()
plt.show()
"""

# TODO: 포트폴리오 가치 평가

#? 옵션 포지션 모형

me_dax = MarketEnvironment('me_dax', pricing_date)
me_dax.add_constant('initial_value', initial_value)
me_dax.add_constant('final_date', pricing_date)
me_dax.add_constant('currency', 'EUR')

me_dax.add_constant('volatility', opt_local[0])
me_dax.add_constant('lambda', opt_local[1])
me_dax.add_constant('mu', opt_local[2])
me_dax.add_constant('delta', opt_local[3])

me_dax.add_constant('model', 'jd')

payoff_func = 'np.maximum(strike - instrument_values, 0)'

shared_env = MarketEnvironment('shared_env', pricing_date)
shared_env.add_constant('maturity', maturity)
shared_env.add_constant('currency', 'EUR')

option_positions = {}
option_environment = {}
for option in option_selection.index:
  option_environment[option] = MarketEnvironment('am_put_%d' % option, pricing_date)
  strike = option_selection['STRIKE_PRC'].loc[option]
  option_environment[option].add_constant('strike', strike)
  option_environment[option].add_environment(shared_env)
  option_positions['am_put_%d' % strike] = DerivativesPosition('am_put_%d' % strike,
                                                               quantity=np.random.randint(10, 50),
                                                               underlying='dax_model',
                                                               market_env=option_environment[option],
                                                               otype='American',
                                                               payoff_func=payoff_func)
  

#? 옵션 포트폴리오
val_env = MarketEnvironment('val_env', pricing_date)
val_env.add_constant('starting_date', pricing_date)
val_env.add_constant('final_date', pricing_date)
val_env.add_constant('frequency', 'B')
val_env.add_constant('paths', 25000)
val_env.add_curve('discount_curve', csr)

underlyings = {'dax_model' : me_dax}

portfolio = DerivativesPortfolio('portfolio', option_positions, val_env, underlyings)

results = portfolio.get_statistics(fixed_seed=True)

print(results.round(1))

print('-' * 70)

print(results[['pos_value', 'pos_delta', 'pos_vega']].sum().round(1))