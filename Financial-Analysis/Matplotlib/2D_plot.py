import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(1000)

# TODO: 1차원 데이터 표현
y = np.random.standard_normal(20).cumsum() #* 1차원 데이터

plt.figure(figsize=(10, 6)) #? 그림 크기 설정
plt.plot(y, 'b', lw=1.5) #? 두께 1.5의 파란선 설정
plt.plot(y, 'ro') #? 붉은 원으로 데이터 표현
plt.xlabel('index') #? x축 레이블
plt.ylabel('value') #? y축 레이블
plt.title('Simple Plot') #? 제목
plt.show()

# TODO: 2차원 데이터 표현
y = np.random.standard_normal((20, 2)).cumsum(axis=0) #* 2차원 데이터

plt.figure(figsize=(10, 6)) #* 그림 크기 설정
plt.plot(y[:, 0], 'b', lw=1.5, label="1st") #? 1번째 데이터 표현
plt.plot(y[:, 1], 'g', lw=1.5, label="2st") #? 2번째 데이터 표현
plt.plot(y, 'ro') #* 붉은 원으로 데이터 표현
plt.legend(loc=0) #? 레이블 정보 표시
plt.xlabel('index') #* x축 레이블
plt.ylabel('value') #* y축 레이블
plt.title('Simple Plot') #* 제목
plt.show()

# TODO: 크기 차이가 큰 2차원 데이터 표현
y[:, 0] = y[:, 0] + 100 #* 데이터 조정

#? 2개의 y축 활용
fig, ax1 = plt.subplots() #? 1번째 객체 생성
plt.plot(y[:, 0], 'b', lw=1.5, label="1st") #* 1번째 데이터 표현
plt.plot(y[:, 0], 'ro') #* 붉은 원으로 데이터 표현
plt.legend(loc=2) #* 레이블 정보 표시
plt.xlabel('index') #* x축 레이블
plt.ylabel('value 1st') #* y축 레이블
plt.title('Simple Plot') #* 제목

ax2 = ax1.twinx() #? x축을 공유하는 2번째 객체 생성
plt.plot(y[:, 1], 'g', lw=1.5, label="2st") #* 1번째 데이터 표현
plt.plot(y[:, 1], 'ro') #* 붉은 원으로 데이터 표현
plt.legend(loc=0) #* 레이블 정보 표시
plt.ylabel('value 2st') #* y축 레이블
plt.show()

#? 2개 서브플롯 활용
plt.figure(figsize=(10, 6)) #* 그림 크기 설정
plt.subplot(211) #? 위에 서브플롯 1 정의
plt.plot(y[:, 0], 'b', lw=1.5, label="1st") #* 1번째 데이터 표현
plt.plot(y[:, 0], 'ro') #* 붉은 원으로 데이터 표현
plt.legend(loc=2) #* 레이블 정보 표시
plt.ylabel('value') #* y축 레이블
plt.title('Simple Plot') #* 제목
plt.subplot(212) #? 아래에 서브플롯 2 정의
plt.plot(y[:, 1], 'g', lw=1.5, label="1st") #* 1번째 데이터 표현
plt.plot(y[:, 1], 'ro') #* 붉은 원으로 데이터 표현
plt.legend(loc=0) #* 레이블 정보 표시
plt.xlabel('index') #* x축 레이블
plt.ylabel('value') #* y축 레이블
plt.show()

# TODO: 산점도
y = np.random.standard_normal((1000, 2))

plt.figure(figsize=(10, 6)) #* 그림 크기 설정
plt.scatter(y[:, 0], y[:, 1], marker='o') #? 산점도 그래프 생성
plt.xlabel('1st') #* x축 레이블
plt.ylabel('2nd') #* y축 레이블
plt.title('Scatter Plot') #* 제목
plt.show()

# TODO: 히스토그램
y = np.random.standard_normal((1000, 2))

plt.figure(figsize=(10, 6)) #* 그림 크기 설정
plt.hist(y, label=['1st', '2nd'], color=['b', 'g'], bins=20) #? 히스토그램 그래프 생성
#? stacked = True를 추가하면 히스토그램 2개가 겹친다.
plt.xlabel('value') #* x축 레이블
plt.ylabel('frequency') #* y축 레이블
plt.title('Histogram Plot') #* 제목
plt.show()

# TODO: 박스 플롯
y = np.random.standard_normal((1000, 2))

fig, ax = plt.subplots(figsize=(10, 6)) #* 1번째 객체 생성
plt.boxplot(y) #? 박스 플롯 생성
plt.setp(ax, xticklabels=['1st', '2nd']) #? 개별 x 레이블 생성
plt.xlabel('data set') #* x축 레이블
plt.ylabel('value') #* y축 레이블
plt.title('Boxplot Plot') #* 제목
plt.show()
