import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# TODO: 옵션 데이터

#? 데이터 읽기
dax = pd.read_csv('data/tr_eikon_option_data.csv')

#? 날짜 컬럼 형식 변환
for col in ['CF_DATE', 'EXPIR_DATE']:
    dax[col] = dax[col].apply(lambda date: pd.Timestamp(date))

#? 초기 가격 설정
initial_value = dax.iloc[0]['CF_CLOSE']

#? Call과 Put 옵션 분리
calls = dax[dax['PUTCALLIND'].str.strip() == 'CALL'].copy()
puts = dax[dax['PUTCALLIND'].str.strip() == 'PUT'].copy()

#? 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1 = calls.set_index('STRIKE_PRC')[['CF_CLOSE', 'IMP_VOLT']].plot(secondary_y='IMP_VOLT', style=['bo', 'rv'], ax=ax1)
ax1.set_title('Call Options')
ax1.set_xlabel('Strike Price')
ax1.set_ylabel('Close Price')
ax1.right_ax.set_ylabel('Implied Volatility')

ax2 = puts.set_index('STRIKE_PRC')[['CF_CLOSE', 'IMP_VOLT']].plot(secondary_y='IMP_VOLT', style=['bo', 'rv'], ax=ax2)
ax2.set_title('Put Options')
ax2.set_xlabel('Strike Price')
ax2.set_ylabel('Close Price')
ax2.right_ax.set_ylabel('Implied Volatility')

plt.tight_layout()
plt.show()
