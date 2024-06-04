import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from Valuation.market_environment import MarketEnvironment
from Valuation.constant_short_rate import ConstantShortRate
from Simulation.square_root_diffusion import SquareRootDiffusion

#? 시장 환경 설정
env = MarketEnvironment('env', dt.datetime(2020, 1, 1))
env.add_constant('initial_value', .25)
env.add_constant('volatility', 0.05)
env.add_constant('final_date', dt.datetime(2020, 12, 31))
env.add_constant('currency', 'EUR')
env.add_constant('frequency', 'W')
env.add_constant('paths', 10000)
env.add_constant('kappa', 4.0)
env.add_constant('theta', 0.2)

#? 할인율 곡선 설정
csr = ConstantShortRate('r', 0.0)
env.add_curve('discount_curve', csr)

#? GBM 시뮬레이션 객체 생성 및 시간 그리드 생성
srd = SquareRootDiffusion('srd', env)

paths = srd.get_instrument_values()[:, :10]

#? 시각화
plt.figure(figsize=(10, 6))
plt.plot(srd.time_grid, paths)
plt.axhline(env.get_constant('theta'), color='r', ls='--', lw=2.0)
plt.xticks(rotation=30)
plt.show()
