import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: 빈도주의 방법론

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

#? 이진화된 포지션 값 생성
def create_bins(data, bins=[0]):
  global cols_bin
  cols_bin = []  #* 이진화된 컬럼 이름 저장 리스트
  for col in lag_cols:
    col_bin = col + '_bin'  #* 이진화된 컬럼 이름 생성
    data[col_bin] = np.digitize(data[col], bins=bins)  #* 이진화 계산
    cols_bin.append(col_bin)  #* 컬럼 이름 리스트에 추가

create_bins(data)

#? 포지션 설정
data['position_frequency'] = np.where(data[cols_bin].sum(axis=1) == 2, -1, 1)  #* 이진화된 컬럼의 합이 2이면 매도(-1), 아니면 매수(1)

#? 전략 수익률 계산
data['strategy_frequency'] = data['position_frequency'] * data['log_returns']

#? 누적 수익률 시각화
data[['log_returns', 'strategy_frequency']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.title('Cumulative Returns for Frequency Strategy')  #* 그래프 제목 설정
plt.xlabel('Date')  #* x축 라벨 설정
plt.ylabel('Cumulative Returns')  #* y축 라벨 설정
plt.show()  #* 그래프 출력
