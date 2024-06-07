import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from Environment.market_environment import MarketEnvironment
from Utils.constant_short_rate import ConstantShortRate
from Simulation.geometric_brownian_motion import GeometricBrownianMotion
from Valuation.mcs_american import MCSAmerican
from Utils.plot_option_stats import plot_option_stats
from Utils.derivatives_position import DerivativesPosition

#? 시장 환경 설정
ME_GBM = MarketEnvironment('ME_GBM', dt.datetime(2020, 1, 1))
ME_GBM.add_constant('initial_value', 36.)
ME_GBM.add_constant('volatility', 0.2)
ME_GBM.add_constant('currency', 'EUR')
ME_GBM.add_constant('model', 'gbm')

ME_AM_PUT = MarketEnvironment('ME_AM_PUT', dt.datetime(2020, 1, 1))
ME_AM_PUT.add_constant('maturity', dt.datetime(2020, 12, 31))
ME_AM_PUT.add_constant('strike', 40.)
ME_AM_PUT.add_constant('currency', 'EUR')

payoff_func = 'np.maximum(strike - instrument_values, 0)'

AM_PUT_POS = DerivativesPosition(name='AM_PUT_POS', quantity=3, underlying='gbm', market_env=ME_AM_PUT, otype='American', payoff_func=payoff_func)

print(AM_PUT_POS.get_info())