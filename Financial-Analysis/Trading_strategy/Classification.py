import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#? 분류 알고리즘 (함수 정의)
#? 랜덤워크 (RWH) 지연값 생성
def create_lags(data):
  global lag_cols
  lag_cols = []
  for lag in range(1, lags + 1):
    col_name = f'lag_{lag}'
    data[col_name] = data['log_returns'].shift(lag)  #* 지연값 계산
    lag_cols.append(col_name)

#? 이진화된 포지션 값 생성
def create_bins(data, bins=[0]):
  global cols_bin
  cols_bin = []
  for col in lag_cols:
    col_bin = col + '_bin'
    data[col_bin] = np.digitize(data[col], bins=bins)  #* 이진화 계산
    cols_bin.append(col_bin)

#? 모델 학습
def fit_models(train_data):
  for model in models.keys():
    models[model].fit(train_data[cols_bin], train_data['direction'])

#? 학습된 모델로 예측하여 포지션 설정
def derive_positions(test_data):
  for model in models.keys():
    test_data['position_' + model] = models[model].predict(test_data[cols_bin])

#? 전략 평가
def evaluate_strategies(data):
  global strategy_returns
  strategy_returns = []
  for model in models.keys():
    col = 'strategy_' + model
    data[col] = data['position_' + model] * data['log_returns'] #* 해당 전략의 수익률 계산
    strategy_returns.append(col) 
  strategy_returns.insert(0, 'log_returns') #* 로그 수익율을 맨 앞에 추가
  strategy_returns.append('strategy_frequency') #* 빈도주의 수익률 추가

#? 데이터 준비
raw_data = pd.read_csv('data/tr_eikon_eod_data.csv').dropna() #* CSV 파일에서 데이터 로드 및 결측값 제거
symbol = 'EUR=' #* 분석할 종목 심볼 설정
data = pd.DataFrame(raw_data[symbol]) #* 선택한 종목 데이터프레임으로 변환

#? 로그 수익률 계산
data['log_returns'] = np.log(data / data.shift(1)) #* 로그 수익률 계산
data.dropna(inplace=True) #* 결측값 제거

#? 수익률 방향 계산
data['direction'] = np.sign(data['log_returns']).astype(int) #* 수익률 방향 계산 (-1, 0, 1)

lags = 5
create_lags(data)
data.dropna(inplace=True)
create_bins(data)
data.dropna(inplace=True)

#? 포지션 설정
data['position_frequency'] = np.where(data[cols_bin].sum(axis=1) == 2, -1, 1)  #* 이진화된 컬럼의 합이 2이면 매도(-1), 아니면 매수(1)
data.dropna(inplace=True)

train_data, test_data = train_test_split(data, test_size=0.5, shuffle=True, random_state=100)

#? 전략 수익률 계산
data['strategy_frequency'] = data['position_frequency'] * data['log_returns']

#? 모델 정의
C = 1
models = {
  'Logistic': LogisticRegression(C=C),
  'Gaussian': GaussianNB(),
  'SVM': SVC(C=C)
}

fit_models(train_data)
derive_positions(data)
evaluate_strategies(data)

#? 각 전략별 누적 수익률 계산
strategy_cum_returns = data[strategy_returns].cumsum().apply(np.exp)

#? 각 전략별 누적 수익률 시각화
print(data[strategy_returns].sum().apply(np.exp))
strategy_cum_returns.plot(figsize=(10, 6))
plt.title('Cumulative Returns of Trading Strategies')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
