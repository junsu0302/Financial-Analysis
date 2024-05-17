import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = 'data/tr_eikon_eod_data.csv'
data = pd.read_csv(filename, index_col=0, parse_dates=True)
"""
#? index_col : 해당 열을 인덱스로 설정
#? parse_dates : 인덱스 값을 날짜 및 시간으로 인삭
"""

data.plot(figsize=(10,12), subplots=False) #? 데이터 시각화
""" 
#* AAPL.O | Apple Stock
#* MSFT.O | Microsoft Stock
#* INTC.O | Intel Stock
#* AMZN.O | Amazon Stock
#* GS.N   | Goldman Sachs Stock
#* SPY    | SPDR S&P 500 ETF Trust
#* .SPY   | S&P 500 Index
#* .VIX   | VIX Volatility Index
#* EUR=   | EUR/USD Exchange Rate
#* XAU=   | Gold Price
#* GDX    | VanEck Vectors Gold Miners ETF
#* GLD    | SPDR Gold Trust
"""

data.describe() #? 데이터의 기본 통계 확인
data.aggregate([min, np.mean, np.std, np.median, max]) #? 사용자 정의 통계 확인

# TODO: 데이터의 시간에 따른 변화 (절대값, 퍼센트값, 로그 수익률)

#* 통계 분석에서는 절대적인 값보다 시간에 따른 변화에 기반하는 경우가 많다.
#* 절대값 < 퍼센트 < 로그 수익률 (계산이 가능)
data.diff() #? 두 인덱스 값 사이의 차이 반환 (절대값)
data.pct_change() #? 두 인덱스 값 사이의 차이 반환 (퍼센트)

rets = np.log(data / data.shift(1)) #? 로그 수익률
rets.cumsum().apply(np.exp).plot(figsize=(10,12)) #? 시간에 따른 누적 로그 수익률

# TODO: 리샘플링
data.resample('1w', label='right').last() #? 주간 데이터로 리샘플링
data.resample('1m', label='right').last() #? 월간 데이터로 리샘플링
data.resample('1y', label='right').last() #? 연간 데이터로 리샘플링

rets.cumsum().apply(np.exp).resample('1y', label='right').last().plot(figsize=(10,12))

# TODO: 기술적 분석(이동 통계)
sym = 'AAPL.O'
data = pd.DataFrame(data[sym]).dropna() #* AAPL.O로만 데이터 준비

window = 20 #? 사용할 데이터 개수 정의
data['min'] = data[sym].rolling(window=window).min() #? 이동 최솟값 계산
data['mean'] = data[sym].rolling(window=window).mean() #? 이동 평균 계산
data['std'] = data[sym].rolling(window=window).std() #? 이동 표준편차 계산
data['median'] = data[sym].rolling(window=window).median() #? 이동 중앙값 계산
data['max'] = data[sym].rolling(window=window).max() #? 이동 최댓값 계산
data['ewma'] = data[sym].ewm(halflife=0.5, min_periods=window).mean() #? 반감기가 0.5인 지수가중 이동평균 계산
data.dropna() #? 이동 통계 확인

ax = data[['min', 'mean', 'max']].iloc[-200:].plot(figsize=(10,6),
                                                   style=['g--', 'y--', 'r--'],
                                                   lw = 0.8) #? 3가지 이동 통계
data[sym].iloc[-200:].plot(ax=ax, lw=2.0) #? 원본 데이터

# TODO: 이동평균 매매법
#! 단기 이동평균선이 장기 이동평균선보다 올라갈 때 매수, 반대일 때 매도

data['SMA1'] = data[sym].rolling(window=42).mean() #? 단기 이동평균선 계산
data['SMA2'] = data[sym].rolling(window=252).mean() #? 장기 이동평균선 계산

data.dropna(inplace=True) #? 완전한 데이터만 보존
data['positions'] = np.where(data['SMA1'] > data['SMA2'], 1, -1) #? 이동평균 매매법
ax = data[[sym, 'SMA1', 'SMA2', 'positions']].plot(figsize=(10,6),
                                                   secondary_y='positions')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))