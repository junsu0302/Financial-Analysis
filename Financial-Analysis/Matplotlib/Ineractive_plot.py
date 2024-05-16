import numpy as np
import pandas as pd
import cufflinks as cf #! pandas Dataframe의 시각화 라이브러리
import plotly.graph_objs as go #! 상호작용 가능한 그래픽 라이브러리
from plotly._subplots import make_subplots

#? 데이터 불러오기
data = pd.read_csv("data/fxcm_eur_usd_eod_data.csv", index_col=0, parse_dates=True)

#? 최근 60일 데이터 선택
quotes = data.iloc[-60:]

# TODO: 그래프 생성
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])

# TODO: OHLC 그래프 생성
#? OHLC 그래프 추가
fig.add_trace(go.Candlestick(x=quotes.index,
                             open=quotes['OpenAsk'],
                             high=quotes['HighAsk'],
                             low=quotes['LowAsk'],
                             close=quotes['CloseAsk'],
                             name='EUR/USD'),
                             row=1, col=1)

# TODO: Bollinger Bands 그래프 생성
#? 이동평균 계산 (20일 이동평균을 사용)
rolling_mean = quotes['CloseAsk'].rolling(window=20).mean()

#? 이동평균의 표준편차 계산
rolling_std = quotes['CloseAsk'].rolling(window=20).std()

#? 볼린저 밴드의 상한선과 하한선 계산
upper_band = rolling_mean + 2 * rolling_std
middle_band = rolling_mean
lower_band = rolling_mean - 2 * rolling_std

#? 볼린저 밴드 그래프 추가
fig.add_trace(go.Scatter(x=quotes.index, y=upper_band, mode='lines', name='Upper Band', showlegend=False, line=dict(color='blue', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=quotes.index, y=middle_band, mode='lines', name='Bollinger Bands', line=dict(color='blue', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=quotes.index, y=lower_band, mode='lines', name='Lower Band', showlegend=False, line=dict(color='blue', width=1)), row=1, col=1)

# TODO: RSI 그래프 생성
#? RSI 계산
delta = quotes['CloseAsk'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

#? RSI 그래프 추가
fig.add_trace(go.Scatter(x=quotes.index, y=rsi, mode='lines', name='RSI', line=dict(color='green', width=1)), row=2, col=1)


# TODO: 레이아웃 설정 및 그래프 출력
#? 레이아웃 설정
fig.update_layout(title="EUR/USD Exchange Rate (OHLC)", 
                  xaxis_title="Date",
                  yaxis_title="Price",
                  xaxis_rangeslider_visible=False)

#? 그래프 출력
fig.show()