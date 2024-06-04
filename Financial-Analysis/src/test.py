import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from Valuation.market_environment import MarketEnvironment
from Valuation.constant_short_rate import ConstantShortRate
from Simulation.geometric_brownian_motion import GeometricBrownianMotion
from Simulation.jump_diffusion import JumpDiffusion

#? 시장 환경 설정
env = MarketEnvironment('env', dt.datetime(2020, 1, 1))
env.add_constant('initial_value', 36.)
env.add_constant('volatility', 0.2)
env.add_constant('final_date', dt.datetime(2020, 12, 31))
env.add_constant('currency', 'EUR')
env.add_constant('frequency', 'M')
env.add_constant('paths', 10000)
env.add_constant('lambda', 0.3)
env.add_constant('mu', -0.75)
env.add_constant('delta', 0.1)

#? 할인율 곡선 설정
csr = ConstantShortRate('csr', 0.06)
env.add_curve('discount_curve', csr)

#? GBM 시뮬레이션 객체 생성 및 시간 그리드 생성
gbm = GeometricBrownianMotion('gbm', env)
gbm.generate_time_grid()

#? JB 시뮬레이션 객체 생성
jd = JumpDiffusion('jd', env)

#? 낮은 점프 강도 시뮬레이션
paths1 = jd.get_instrument_values()

#? 높은 점프 강도 시뮬레이션
jd.update(lamb=0.9)
paths2 = jd.get_instrument_values()

#? 시각화
plt.figure(figsize=(10, 6))
p1 = plt.plot(gbm.time_grid, paths1[:, :10], 'b')
p2 = plt.plot(gbm.time_grid, paths2[:, :10], 'r-.')
l1 = plt.legend([p1[0], p2[0]], ['low volatility', 'high volatility'], loc=2)
plt.gca().add_artist(l1)
plt.xticks(rotation=30)
plt.show()
"""
변동성이 자산 가격 경로에 미치는 영향이 크다는 것을 확인할 수 있다.
변동성이 높을수록 가격의 변동 폭이 커지며, 이는 투자자에게 더 큰 리스크와 더 큰 기회를 제공한다.
이 시뮬레이션은 옵션 가격 결정, 리스크 관리, 포트폴리오 최적화 등에 활용될 수 있다.
"""