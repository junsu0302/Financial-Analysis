import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from Environment.market_environment import MarketEnvironment
from Utils.constant_short_rate import ConstantShortRate
from Simulation.geometric_brownian_motion import GeometricBrownianMotion
from Valuation.mcs_american import MCSAmerican
from Utils.plot_option_stats import plot_option_stats

#? 시장 환경 설정
env = MarketEnvironment('env', dt.datetime(2020, 1, 1))
env.add_constant('initial_value', 36.) #* 초기 자산 가격
env.add_constant('volatility', 0.2) #* 자산 가격 변동성
env.add_constant('final_date', dt.datetime(2020, 12, 31)) #* 시뮬레이션 종료 날짜
env.add_constant('currency', 'EUR') #* 통화
env.add_constant('frequency', 'W') #* 시뮬레이션 빈도
env.add_constant('paths', 50000) #* 시뮬레이션 경로 수

#? 할인 곡선 생성 및 추가
csr = ConstantShortRate('csr', 0.06)
env.add_curve('discount_curve', csr)

#? 기하 브라운 운동 시뮬레이션 객체 생성
gbm = GeometricBrownianMotion('gbm', env)

#? 옵션 시장 환경 설정
put = MarketEnvironment('put', env.pricing_date)
put.add_constant('strike', 40.)
put.add_constant('maturity', dt.datetime(2020, 12, 31))
put.add_constant('currency', 'EUR')

#? 옵션 페이오프 설정
payoff_func = 'np.maximum(strike - instrument_values, 0)'

#? 유럽형 옵션 평가 객체 생성
am_put = MCSAmerican('am_put', underlying=gbm, market_env=put, payoff_func=payoff_func)

am_put.present_value(fixed_seed=True, bf=5)

ls_table = []
for initial_value in (36., 38., 40., 42., 44.):
  for volatility in (0.2, 0.4):
    for maturity in (dt.datetime(2020, 12, 31), dt.datetime(2021, 12, 31)):
      am_put.update(initial_value=initial_value, volatility=volatility, maturity=maturity)
      ls_table.append([initial_value, volatility, maturity, am_put.present_value(bf=5)])

print('S0 | Vola | T | Value')
print(22 * '-')
for r in ls_table:
  print('%d | %3.1f | %d | %5.3f' % (r[0], r[1], r[2].year - 2019, r[3]))