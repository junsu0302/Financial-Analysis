import numpy as np
import pandas as pd

import pickle #! 객체 직렬화 (객체 -> 바이트열)
from random import gauss #* 정규분포 난수 생성


# TODO: 객체를 직렬화하여 디스크에 저장 및 디스크에서 객체 읽기
a = [gauss(1.5, 2) for i in range(1000000)] #* 난수 리스트 생성

path = 'cache/' #? 데이터를 저장할 파일 경로 지정
pkl_file = open(path + 'data.pkl', 'wb') #? 파일을 바이너리 쓰기 모드로 열기
pickle.dump(a, pkl_file) #? 객체를 직렬화하여 파일에 저장
pkl_file.close() #? 파일 닫기

pkl_file = open(path + 'data.pkl', 'rb') #? 파일을 바이너리 읽기 모드로 열기
data = pickle.load(pkl_file) #? 디스크에서 객체를 읽어 역직렬화

# TODO: CSV 읽기 및 쓰기
a = np.random.standard_normal((5000, 5)) #* 난수로 이루어진 객체 생성
t = pd.date_range(start='2000/1/1', periods=5000, freq='H') #* 시간 단위 시계열 객체 생성

path = 'cache/' #? 데이터를 저장할 파일 경로 지정
csv_file = open(path + 'data.csv', 'w') #? 파일을 쓰기 모드로 열기
header = 'data, no1, no2, no3, no4, no5\n' #? 헤더 행을 정의
csv_file.write(header) #? 첫 줄에 헤더 행 쓰기

#? 행 단위로 데이터를 조합하여 쓰기
for t_, (no1, no2, no3, no4, no5) in zip(t, a):
  s = '{}, {}, {}, {}, {}\n'.format(t_, no1, no2, no3, no4, no5)
  csv_file.write(s)

csv_file.close() #? 파일 닫기

import csv

with open(path + 'data.csv', 'r') as f: #? 파일을 읽기 모드로 열기
  csv_reader = csv.reader(f) #? 각 줄을 리스트 객체로 변환
  lines = [line for line in csv_reader]
csv_file.close() #? 파일 닫기

# TODO: SQL DB 읽기 및 쓰기
import sqlite3 as sq3
import datetime

path = 'cache/' #? 데이터를 저장할 파일 경로 지정
con = sq3.connect(path + 'numbs.db') #? DB 연결
query = 'CREATE TABLE numbs (Date date, No1 real, No2 real)' #? 3개 열을 갖는 테이블 생성

con.execute(query) #? 쿼리 실행
con.commit() #? 커밋

q = con.execute
q('SELECT * FROM sqlite_master').fetchall() #? DB 메타정보 읽기

#* 난수 데이터 생성
np.random.seed(100)
data = np.random.standard_normal((10000, 2)) 

#? DB에 데이터 삽입
for row in data:
  now = datetime.datetime.now()
  q('INSERT INTO numbs VALUES(?, ?, ?)', (now, row[0], row[1])) #? numbs 테이블에 한 행 쓰기
con.commit()

#? DB 데이터 반환
q('SELECT * FROM numbs').fetchone() #? 조건에 맞는 데이터 1개 반환
q('SELECT * FROM numbs').fetchmany(5) #? 조건에 맞는 데이터 '5'개 반환
q('SELECT * FROM numbs').fetchall() #? 조건에 맞는 모든 데이터 반환

#? DB 삭제
q('DROP TABLE IF EXISTS numbs') #? DB에서 해당 테이블 제거

con.close() #? DB 연결 닫기