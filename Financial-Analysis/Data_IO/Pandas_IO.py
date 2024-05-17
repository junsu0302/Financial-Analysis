import numpy as np
import pandas as pd
import sqlite3 as sq3
import matplotlib.pyplot as plt

# TODO: DB에 데이터 저장
data = np.random.standard_normal((1000000, 5))

path = 'cache/'
con = sq3.Connection(path + 'numbers.db')
query = 'CREATE TABLE numbers (No1 real, No2 real, No3 real, No4 real, No5 real)'

#? 명령어 정의
q = con.execute
qm = con.executemany

# q(query) #? 5개의 열이 있는 테이블 생성
# qm('INSERT INTO numbers VALUES (?, ?, ?, ?, ?)', data) #? 데이터 입력
# con.commit() #? 변경점 반영

# TODO: 데이터 쿼리 및 시각화
data  = q('SELECT * FROM numbers').fetchall() #? 전체 데이터 반환
query = 'SELECT * FROM numbers WHERE No1 > 0 AND No2 < 0'
res = np.array(q(query).fetchall()) #? 조건에 맞는 모든 데이터 반환

res = res[::100] #? 데이터의 일부만 사용 (가독성)
# plt.figure(figsize=(10,6))
# plt.plot(res[:, 0], res[:, 1], 'ro')

# TODO: Pandas로 데이터 쿼리
data = pd.read_sql('SELECT * FROM numbers', con) #? 테이블의 모든 데이터를 DataFrame 객체에 반환

q = '(No1 < -0.5 | No1 > 0.5) & (No2 < -1 | No2 > 1)' #? 쿼리 조건
res = data[['No1', 'No2']].query(q) #? 조건에 맞는 모든 데이터 반환
plt.figure(figsize=(10,6))
plt.plot(res['No1'], res['No2'], 'ro')

# TODO: CSV 파일로 관리
data.to_csv(path + 'numbers.csv') #? CSV 형식으로 파일 저장
data = pd.read_csv(path + 'numbers.csv') #? CSV 파일 읽기 