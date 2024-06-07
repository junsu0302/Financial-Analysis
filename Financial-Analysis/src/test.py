import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from Environment.market_environment import MarketEnvironment
from Utils.constant_short_rate import ConstantShortRate
from Simulation.geometric_brownian_motion import GeometricBrownianMotion
from Valuation.mcs_european import MCSEuropean
from Utils.plot_option_stats import plot_option_stats

#? 시장 환경 설정
env = MarketEnvironment('env', dt.datetime(2020, 1, 1))
env.add_constant('initial_value', 36.) #* 초기 자산 가격
env.add_constant('volatility', 0.2) #* 자산 가격 변동성
env.add_constant('final_date', dt.datetime(2020, 12, 31)) #* 시뮬레이션 종료 날짜
env.add_constant('currency', 'EUR') #* 통화
env.add_constant('frequency', 'M') #* 시뮬레이션 빈도
env.add_constant('paths', 10000) #* 시뮬레이션 경로 수

#? 할인 곡선 생성 및 추가
csr = ConstantShortRate('csr', 0.06)
env.add_curve('discount_curve', csr)

#? 기하 브라운 운동 시뮬레이션 객체 생성
gbm = GeometricBrownianMotion('gbm', env)

#? 옵션 시장 환경 설정
call = MarketEnvironment('call', env.pricing_date)
call.add_constant('strike', 40.)
call.add_constant('maturity', dt.datetime(2020, 12, 31))
call.add_constant('currency', 'EUR')

#? 옵션 페이오프 설정
payoff_func = 'np.maximum(maturity_value - strike, 0)'

#? 유럽형 옵션 평가 객체 생성
eur_call = MCSEuropean('eur_call', underlying=gbm, market_env=call, payoff_func=payoff_func)

#? 옵션의 현재 가치, 델타, 베가 계산
eur_call.present_value()
eur_call.delta()
eur_call.vega()

#? 다양한 기초 자산 가격에 대해 옵션 가격, 델타, 베가 계산
s_list = np.arange(34., 46.1, 2.) #* 기초 자산 가격
p_list = [] #* 기초 자산 가격에 대한 옵션의 현재 가치
d_list = [] #* 기초 자산 가격에 대한 옵션의 델타 값
v_list = [] #* 기초 자산 가격에 대한 옵션의 베가 값
for s in s_list:
  eur_call.update(initial_value=s)
  p_list.append(eur_call.present_value(fixed_seed=True))
  d_list.append(eur_call.delta())
  v_list.append(eur_call.vega())

#? 시각화
plot_option_stats(s_list, p_list, d_list, v_list)
