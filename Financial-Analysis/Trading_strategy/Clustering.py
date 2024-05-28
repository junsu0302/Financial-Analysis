import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# TODO: 클러스터링

#? 데이터 준비
raw_data = pd.read_csv('data/tr_eikon_eod_data.csv').dropna()  #* CSV 파일에서 데이터 로드 및 결측값 제거
symbol = 'EUR='  #* 분석할 종목 심볼 설정
data = pd.DataFrame(raw_data[symbol])  #* 선택한 종목 데이터프레임으로 변환

#? 로그 수익률 계산
data['log_returns'] = np.log(data / data.shift(1))  #* 로그 수익률 계산
data.dropna(inplace=True)  #* 결측값 제거

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

#? KMeans 클러스터링 모델 생성 및 학습
model = KMeans(n_clusters=2, random_state=0) #* 2개의 클러스터 생성
model.fit(data[lag_cols]) #* 모델 학습
data['position_clustering'] = model.predict(data[lag_cols]) #* 클러스터 예측값 저장
data['position_clustering'] = np.where(data['position_clustering'] == 1, -1, 1) #* 클러스터링 결과를 매수/매도 포지션으로 변환

#? 전략 수익률 계산
data['strategy_clustering'] = data['position_clustering'] * data['log_returns']

#? 누적 수익률 계산
cumulative_returns = data[['log_returns', 'strategy_clustering']].cumsum().apply(np.exp)  #* 누적 수익률 계산

#? 시각화
cumulative_returns.plot(figsize=(10, 6))  #* 누적 수익률 그래프 그리기
plt.title('Cumulative Returns for Market and Clustering Strategy')  #* 그래프 제목
plt.xlabel('Date')  #* x축 라벨
plt.ylabel('Cumulative Returns')  #* y축 라벨
plt.legend(['Market Returns', 'Clustering Strategy Returns'])  #* 범례 설정
plt.show()  #* 그래프 출력