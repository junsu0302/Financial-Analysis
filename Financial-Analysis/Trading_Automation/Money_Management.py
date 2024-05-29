import math
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# TODO: 캘리 기준 전략
"""
캘리 전략은 자산을 어떻게 배분해야 할지를 정하는 수학적 기법이다.
다음 가정에 따라 최적의 투자 비율을 계산하여 최대의 기대 이익을 얻는걸 목표로 한다.
- 독립적인 시행
- 고정된 배당률
- 투자금의 증가 or 감소
"""

#? 데이터 준비
raw_data = pd.read_csv('data/tr_eikon_eod_data.csv').dropna() #* CSV 파일에서 데이터 로드 및 결측값 제거
symbol = '.SPX' #* 분석할 종목 심볼 설정

#? 날짜 데이터를 datetime 형식으로 변환하고 인덱스로 설정
raw_data['Date'] = pd.to_datetime(raw_data['Date']) #* 'Date' 컬럼을 datetime 형식으로 변환
raw_data.set_index('Date', inplace=True) #* 'Date' 컬럼을 인덱스로 설정

#? 선택한 종목 데이터프레임으로 변환
data = pd.DataFrame(raw_data[symbol])

#? 로그 수익률 계산
data['returns'] = np.log(data / data.shift(1)) #* 로그 수익률 계산
data.dropna(inplace=True) #* 결측값 제거

#? 캘리 비율 계산
mu = data.returns.mean() * 252 #* 연간 기대 수익률 계산
sigma = data.returns.std() * 252 ** 0.5 #* 연간 수익률의 표준편차 계산
r = 0.0 #* 이자율 설정
f = (mu - r) / sigma ** 2 #* 캘리 비율 계산

equity_columns = []
def kelly_strategy(kelly_fraction:float):
  """주어진 데이터셋에 캘리 기준 전략 사용

  Args:
    kelly_fraction (float): 자본 할당을 결정하는 켈리 비율
  """
  global equity_columns
  equity_column = 'equity_kelly_{:.2f}'.format(kelly_fraction) #* 켈리 비율에 따른 자산 이름을 설정
  capital_column = 'capital_kelly_{:.2f}'.format(kelly_fraction) #* 켈리 비율에 따른 자본 이름을 설정
  
  equity_columns.append(equity_column)
  data[equity_column] = 1 #* 자산 컬럼 초기값을 1로 설정
  data[capital_column] = data[equity_column] * kelly_fraction #* 자본 컬럼 초기값을 자산에 켈리 비율을 곱한 값으로 설정

  for i, date in enumerate(data.index[1:]):
    prev_date = data.index[i] #* 이전 날짜
    data.loc[date, capital_column] = data[capital_column].loc[prev_date] * math.exp(data['returns'].loc[date]) #* 자본에 수익률을 지수화하여 현재 자본 업데이트
    data.loc[date, equity_column] = data[capital_column].loc[date] - data[capital_column].loc[prev_date] + data[equity_column].loc[prev_date] #* 자본 변화량을 더하고, 이전 자산을 빼서 현재 자산 업데이트
    data.loc[date, capital_column] = data[equity_column].loc[date] * kelly_fraction #* 현재 자산에 켈리 비율을 곱하여 자본 업데이트

kelly_strategy(f)

#? 결과 시각화
plt.figure(figsize=(10, 6)) 
plt.plot(data['returns'].cumsum().apply(np.exp), label='Returns', color='black', linestyle='dashed') #* 누적 수익률
for i, equ in enumerate(equity_columns):
  plt.plot(data[equ], label=equ, alpha=0.8) #* 각 전략별 자산 그래프
plt.title('Kelly Criterion Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()