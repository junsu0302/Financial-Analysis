import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: 상관관계 분석
#! 일반적으로 S&P 500지수와 VIX는 음의 상관관계를 갖는다.

# TODO: 데이터 준비
filename = 'data/tr_eikon_eod_data.csv'
raw_data = pd.read_csv(filename, index_col=0, parse_dates=True)

data = raw_data[['.SPX', '.VIX']].dropna()

# TODO: 로그 수익률
rets = np.log(data / data.shift(1))
rets.dropna(inplace=True)
# pd.plotting.scatter_matrix(rets, #? 데이터
#                            alpha=0.2, #? 투명도 설정 
#                            diagonal='hist', #? 대각선 부분 그래프
#                            hist_kwds={'bins': 35}, #? 그래프에 넣은 인수 키워드
#                            figsize=(10,6))
#* 시간에 따른 변화를 나타내는 로그 수익률의 경우 주가지수의 변동성이 높은 시기에 변동성 클러스터가 발생한다.
#* 이 떄, 시계열의 로그 수익률을 그리면서 히스토그램이나 커널 밀도 추정치를 그릴 수 있다.

# TODO: OLS 회귀법(최소 자승 회귀법)
#! 두 지수의 선형성 확인
reg = np.polyfit(rets['.SPX'], rets['.VIX'], deg=1) #? OLS 선형회귀 구현

# ax = rets.plot(kind='scatter', x='.SPX', y='.VIX', figsize=(10,6)) #? 로그 수익률 스캐터 플롯
# ax.plot(rets['.SPX'], np.polyval(reg, rets['.SPX']), 'r', lw=2) #? 선형회귀선

# TODO: 상관관계 측정
rets.corr() #? 전체 Dataframe에 대한 상관관계 행렬

ax = rets['.SPX'].rolling(window=252).corr(rets['.VIX']).plot(figsize=(10,6)) #? 시간에 따라 변하는 이동 상관관계 플롯
ax.axhline(rets.corr().iloc[0,1], c='r') #? 수평선으로 전체 상관관계 표시

plt.show()