import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from Environment.market_environment import MarketEnvironment
from Utils.constant_short_rate import ConstantShortRate
from Utils.derivatives_position import DerivativesPosition
from Utils.derivatives_portfolio import DerivativesPortfolio

#? 기하 브라운 운동 (GBM) 모델을 위한 시장 환경 설정
ME_GBM = MarketEnvironment('ME_GBM', dt.datetime(2020, 1, 1))
ME_GBM.add_constant('initial_value', 36.) #* 기초 자산의 초기 가격
ME_GBM.add_constant('volatility', 0.2) #* 기초 자산의 변동성
ME_GBM.add_constant('currency', 'EUR') #* 기초 자산의 통화
ME_GBM.add_constant('model', 'gbm') #* 모델 타입

#? 점프 확산 (JD) 모델을 위한 시장 환경 설정
ME_JD = MarketEnvironment('ME_JD', ME_GBM.pricing_date)
ME_JD.add_constant('lambda', 0.3) #* 점프의 강도
ME_JD.add_constant('mu', -0.75) #* 점프 크기의 평균
ME_JD.add_constant('delta', 0.1) #* 점프 크기의 표준 편차
ME_JD.add_environment(ME_GBM) #* GBM 환경 추가
ME_JD.add_constant('model', 'jd') #* 모델 타입

#? 아메리칸 풋 옵션을 위한 시장 환경 설정
ME_AM_PUT = MarketEnvironment('ME_AM_PUT', dt.datetime(2020, 1, 1))
ME_AM_PUT.add_constant('maturity', dt.datetime(2020, 12, 31)) #* 옵션 만기일
ME_AM_PUT.add_constant('strike', 40.) #* 옵션 행사가
ME_AM_PUT.add_constant('currency', 'EUR') #* 옵션 통화

#? 아메리칸 풋 옵션의 페이오프 함수
payoff_func = 'np.maximum(strike - instrument_values, 0)'

#? 아메리칸 풋 옵션 포지션 생성
AM_PUT_POS = DerivativesPosition(name='AM_PUT_POS', quantity=3, underlying='gbm', market_env=ME_AM_PUT, otype='American', payoff_func=payoff_func)

#? 유럽식 콜 옵션을 위한 시장 환경 설정
ME_EUR_CALL = MarketEnvironment('ME_EUR_CALL', ME_JD.pricing_date)
ME_EUR_CALL.add_constant('maturity', dt.datetime(2020, 6, 30)) #* 옵션 만기일
ME_EUR_CALL.add_constant('strike', 38.) #* 옵션 행사가
ME_EUR_CALL.add_constant('currency', 'EUR') #* 옵션 통화

#? 유럽식 콜 옵션의 페이오프 함수
payoff_func = 'np.maximum(maturity_value - strike, 0)'

#? 유럽식 콜 옵션 포지션 생성
EUR_CALL_POS = DerivativesPosition(name='EUR_CALL_POS', quantity=5, underlying='jd', market_env=ME_EUR_CALL, otype='European', payoff_func=payoff_func)

#? 기초 자산 모델과 포지션 정의
underlyings = {
  'gbm': ME_GBM,
  'jd': ME_JD
}
positions = {
  'AM_PUT_POS': AM_PUT_POS,
  'EUR_CALL_POS': EUR_CALL_POS
}

#? 할인율을 위한 상수 단기 이자율 설정
csr = ConstantShortRate('csr', 0.06)

#? 평가를 위한 일반적인 시장 환경 설정
valuation_env = MarketEnvironment('general', ME_GBM.pricing_date)
valuation_env.add_constant('frequency', 'W') #* 시뮬레이션 단계의 빈도
valuation_env.add_constant('paths', 25000) #* 몬테카를로 시뮬레이션 경로 수
valuation_env.add_constant('starting_date', valuation_env.pricing_date) #* 시뮬레이션 시작일
valuation_env.add_constant('final_date', valuation_env.pricing_date) #* 시뮬레이션 종료일 (단순성을 위해 시작일과 동일)
valuation_env.add_curve('discount_curve', csr) #* 환경에 할인율 곡선 추가

#? 파생상품 포트폴리오 생성
portfolio = DerivativesPortfolio(name='portfolio',
                                 positions=positions,
                                 valuation_env=valuation_env,
                                 assets=underlyings,
                                 fixed_seed=False)

#? 시각화를 위한 특정 경로 선택
path_no = 888
path_gbm = portfolio.underlying_objects['gbm'].get_instrument_values()[:, path_no]
path_jd = portfolio.underlying_objects['jd'].get_instrument_values()[:, path_no]

#? 세 개의 가로형 서브플롯 생성
fig, axs = plt.subplots(1, 3, figsize=(20, 5))

#? 첫 번째 서브플롯: GBM 및 JD 모델 경로 시각화
axs[0].plot(portfolio.time_grid, path_gbm, 'r', label='gbm')
axs[0].plot(portfolio.time_grid, path_jd, 'b', label='jd')
axs[0].set_title('GBM and JD Paths')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Value')
axs[0].legend(loc='best')
axs[0].tick_params(axis='x', rotation=30)

#? 두 번째 서브플롯: 개별 옵션 포지션의 현재 가치 히스토그램
pv1 = 5 * portfolio.valuation_objects['EUR_CALL_POS'].present_value(full=True)[1]
pv2 = 3 * portfolio.valuation_objects['AM_PUT_POS'].present_value(full=True)[1]
axs[1].hist([pv1, pv2], bins=25, label=['European call', 'American put'])
axs[1].axvline(pv1.mean(), color='r', ls='dashed', lw=1.5, label='call mean: %4.2f' % pv1.mean())
axs[1].axvline(pv2.mean(), color='r', ls='dotted', lw=1.5, label='put mean: %4.2f' % pv2.mean())
axs[1].set_title('Histogram of Option Values')
axs[1].set_xlabel('Present Value')
axs[1].set_ylabel('Frequency')
axs[1].set_xlim(0, 80)
axs[1].set_ylim(1, 10000)
axs[1].legend()

#? 세 번째 서브플롯: 포트폴리오 가치 및 표준편차 시각화
portfolio_values = pv1 + pv2
portfolio_std = portfolio_values.std()

axs[2].hist(portfolio_values, bins=50, label='Portfolio Values')
axs[2].axvline(portfolio_values.mean(), color='r', ls='dashed', lw=1.5, label='Mean: %4.2f' % portfolio_values.mean())
axs[2].axvline(portfolio_values.mean() + portfolio_std, color='g', ls='dashed', lw=1.5, label='Mean + 1 Std: %4.2f' % (portfolio_values.mean() + portfolio_std))
axs[2].axvline(portfolio_values.mean() - portfolio_std, color='g', ls='dashed', lw=1.5, label='Mean - 1 Std: %4.2f' % (portfolio_values.mean() - portfolio_std))
axs[2].set_title('Portfolio Values with Std Dev')
axs[2].set_xlabel('Present Value')
axs[2].set_ylabel('Frequency')
axs[2].set_xlim(0, 80)
axs[2].set_ylim(0, 5000)
axs[2].legend()

#? 시각화를 위한 레이아웃 조정
plt.tight_layout()
plt.show()
