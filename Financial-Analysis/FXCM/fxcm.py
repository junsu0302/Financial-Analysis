import time
import fxcmpy
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

api = fxcmpy.fxcmpy(config_file='fxcm.cfg')

# TODO: 데이터 불러오기
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2001, 1, 1)
candles = api.get_candles('USD/JPY', period='D1', start=start, stop=end)
"""
#* bidopen
#* bidclose
#* bidhigh
#* bidlow
#* askopen
#* askclose
#* askhigh
#* asklow
#* tickqty
#? bid price : 매수자가 매수하길 원하는 가격
#? ask price : 매도자가 매도하길 원하는 가격
#? spread : Bid-Ask (시장의 유동성, 효율성)
#? tickqty : 틱 단위로 거래된 수량 (거래량)
"""

# TODO: 계좌정보 조회
api.get_accounts().T
""" 
#* accountId : 계좌 ID
#* accountName : 계좌 이름
#* balance : 잔액
#* dayPL : 일일 손익
#* equity : 총 자산
#* grossPL : 총 손익
#* hedging : 계좌에서 헤징이 활성화되었는지 여부
#* mc : 마진 콜 상태
#* mcDate : 마진 콜 날짜
#* t
#* usableMargin : 사용 가능한 마진
#* usableMargin3
#* usableMargin3Perc
#* usableMarginPerc : 사용 가능한 마진 비율
#* usdMr : USD 마진 요구
#* usdMr3
"""