import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from itertools import product

# TODO: 이동평균 매매전략

#? 데이터 준비
raw = pd.read_csv('data/tr_eikon_eod_data.csv')  #* CSV 파일에서 데이터 로드
symbol = 'AAPL.O'  #* 분석할 종목 심볼 설정
data = pd.DataFrame(raw[symbol])  #* 종목 데이터 선택

#? 최적화를 위한 이동평균 범위 설정
sma1 = range(20, 61, 4)
sma2 = range(180, 281, 10)

results = []

for SMA1, SMA2 in product(sma1, sma2):
  #? 데이터 준비
  data = pd.DataFrame(raw[symbol])
  data.dropna(inplace=True)

  #? 이동평균 계산
  data['SMA1'] = data[symbol].rolling(SMA1).mean()  #* 단기 이동평균 계산
  data['SMA2'] = data[symbol].rolling(SMA2).mean()  #* 장기 이동평균 계산
  data.dropna(inplace=True)  #* 결측값 제거 (NaN 제거)
    
  #? 거래 포지션 설정 (매수 or 매도)
  data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)  #* 단순 이동평균 교차 전략

  #? 벡터화된 백테스팅
  data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))  #* 일간 로그 수익률 계산
  data['Strategy'] = data['Position'].shift(1) * data['Returns']  #* 전략에 따른 수익률 계산
  data.dropna(inplace=True)  #* 결측값 제거 (NaN 제거)

   #? 총 수익률 계산 (누적)
  perf = np.exp(data[['Returns', 'Strategy']].sum()) 

  results.append({
    'SMA1': SMA1, 'SMA2': SMA2,
    'MARKET': perf['Returns'],
    'STRATEGY': perf['Strategy'],
    'OUT': perf['Strategy'] - perf['Returns']
  })

#? 리스트를 데이터프레임으로 변환
results_df = pd.DataFrame(results)  #* 결과 리스트를 데이터프레임으로 변환

#? 전략의 성과를 OUT 기준으로 정렬하여 상위 5개 출력
print(results_df.sort_values('OUT', ascending=False).head()) 

#? 최적화 결과 시각화
best_result = results_df.loc[results_df['OUT'].idxmax()]  #* 최고의 성과를 낸 조합 선택
print(f"Best SMA1: {best_result['SMA1']}, Best SMA2: {best_result['SMA2']}")  #* 최고의 성과를 낸 SMA1과 SMA2 값 출력

data['SMA1'] = data[symbol].rolling(int(best_result['SMA1'])).mean()  #* 최적화된 SMA1 값으로 단기 이동평균 계산
data['SMA2'] = data[symbol].rolling(int(best_result['SMA2'])).mean()  #* 최적화된 SMA2 값으로 장기 이동평균 계산
data.dropna(inplace=True)  #* 결측값 제거 (NaN 제거)

#? 서브플롯 생성 (가로 배치)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  #* 가로로 2개의 서브플롯 생성

#? 첫 번째 서브플롯: 최적화된 이동평균 시각화
ax1.plot(data[symbol], label='Price')  #* 종목 가격 시각화
ax1.plot(data['SMA1'], label=f'SMA1 ({int(best_result["SMA1"])})')  #* 최적화된 단기 이동평균 시각화
ax1.plot(data['SMA2'], label=f'SMA2 ({int(best_result["SMA2"])})')  #* 최적화된 장기 이동평균 시각화
ax1.set_title('Best SMA Strategy')  #* 그래프 제목 설정
ax1.set_xlabel('Date')  #* x축 라벨 설정
ax1.set_ylabel('Price')  #* y축 라벨 설정
ax1.legend(loc='best')  #* 범례 위치 설정

#? 전략 성과 시각화
data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)  #* 최적화된 단기 및 장기 이동평균에 따른 포지션 설정
data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))  #* 일간 로그 수익률 계산
data['Strategy'] = data['Position'].shift(1) * data['Returns']  #* 전략에 따른 수익률 계산
data.dropna(inplace=True)  #* 결측값 제거 (NaN 제거)

#? 두 번째 서브플롯: 전략의 누적 수익률 및 포지션
cumulative_returns = data[['Returns', 'Strategy']].cumsum().apply(np.exp)  #* 누적 수익률 계산
cumulative_returns.plot(ax=ax2)  #* 누적 수익률 시각화
data['Position'].plot(ax=ax2, secondary_y='Position', style='--')  #* 포지션 변화를 시각화
ax2.set_title(f'Cumulative Returns\nBest SMA1: {int(best_result["SMA1"])}, Best SMA2: {int(best_result["SMA2"])}')  #* 그래프 제목 설정
ax2.set_xlabel('Date')  #* x축 라벨 설정
ax2.set_ylabel('Cumulative Returns')  #* y축 라벨 설정
ax2.right_ax.set_ylabel('Position')  #* 오른쪽 y축 라벨 설정
ax2.get_legend().set_bbox_to_anchor((0.25, 0.85))  #* 범례 위치 설정

plt.tight_layout()  #* 레이아웃 조정
plt.show()  #* 그래프 출력