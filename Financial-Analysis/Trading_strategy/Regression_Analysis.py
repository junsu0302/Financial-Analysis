import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# TODO: 선형 회귀분석

#? 데이터 준비
raw_data = pd.read_csv('data/tr_eikon_eod_data.csv').dropna()  #* CSV 파일에서 데이터 로드 및 결측값 제거
symbol = 'EUR='  #* 분석할 종목 심볼 설정
data = pd.DataFrame(raw_data[symbol])  #* 선택한 종목 데이터프레임으로 변환

#? 로그 수익률 계산
data['log_returns'] = np.log(data / data.shift(1))  #* 로그 수익률 계산
data.dropna(inplace=True)  #* 결측값 제거

#? 수익률 방향 계산
data['direction'] = np.sign(data['log_returns']).astype(int)  #* 수익률 방향 계산 (-1, 0, 1)

#? 랜덤워크 (RWH) 지연값 생성
lags = 2  #* 지연값의 개수 설정
def create_lags(data):
  global lag_cols
  lag_cols = []  #* 지연값 컬럼 이름 저장 리스트
  for lag in range(1, lags + 1):
    col_name = f'lag_{lag}'  #* 지연값 컬럼 이름 생성
    data[col_name] = data['log_returns'].shift(lag)  #* 지연값 계산
    lag_cols.append(col_name)  #* 컬럼 이름 리스트에 추가

create_lags(data)
data.dropna(inplace=True)  #* 지연값 계산 후 결측값 제거

#? 회귀분석 모델 생성 및 예측값 계산
linear_model = LinearRegression()
data['position_log_returns'] = linear_model.fit(data[lag_cols], data['log_returns']).predict(data[lag_cols]) #* 로그 수익률을 예측하는 모델 생성
data['position_direction'] = linear_model.fit(data[lag_cols], data['direction']).predict(data[lag_cols]) #* 수익률 방향을 예측하는 모델 생성
data[['position_log_returns', 'position_direction']] = np.where(data[['position_log_returns', 'position_direction']] > 0, 1, -1) #* 예측값을 매수/매도 포지션으로 변환 (0 이상: 매수, 0 미만: 매도)

#? 전략 수익률 계산
data['strategy_log_returns'] = data['position_log_returns'] * data['log_returns']  #* 로그 수익률 전략
data['strategy_direction'] = data['position_direction'] * data['log_returns']  #* 수익률 방향 전략

#? 전략 수익률 시각화
cumulative_returns_plot = data[['log_returns', 'strategy_log_returns', 'strategy_direction']].cumsum().apply(np.exp)
cumulative_returns_plot.plot(figsize=(10, 6))
plt.title('Cumulative Returns for Different Strategies')  #* 그래프 제목
plt.xlabel('Date')  #* x축 라벨
plt.ylabel('Cumulative Returns')  #* y축 라벨
plt.legend(['Market Returns', 'Strategy Log Returns', 'Strategy Direction'])  #* 범례 설정
plt.show()
