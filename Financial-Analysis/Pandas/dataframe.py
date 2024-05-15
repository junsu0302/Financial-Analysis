import pandas as pd

# TODO: 데이터 프레임 정의
df = pd.DataFrame([10, 20, 30, 40], columns=['numbers'], index=['a', 'b', 'c', 'd'])
""" 
#? 레이블이 있는 데이터 생성
#* 데이터를 list 객체로 정의
#* columns : 열 레이블 지정
#* index : 인덱스 레이블 지정
"""

# TODO: 정보 반환
df #? 전체 객체 정보 반환
df.index #? 인덱스 정보(객체, 속성) 반환
df.columns #? 열 정보(객체, 속성) 반환
df.loc['a'] #? 지정된 인덱스 정보 반환
df.iloc[1:3] #? 지정된 영역의 인덱스 정보 반환



# TODO: 데이터 확장
df['floats'] = (0.5, 1.5, 2.5, 3.5) #? 새로운 인덱스 데이터 확장
# df.append({'numbers': 50, 'floats': 4.5}) #? 새로운 데이터 추가

# TODO: 시계열 데이터 관리
import numpy as np
np.random.seed(100)
a = np.random.standard_normal((9, 4))
#* 금융공학에서 ndarray 객체를 사용하여 메타 정보만 활용하는 방법을 많이 사용한다.

df = pd.DataFrame(a) #? ndarray 객체를 활용하여 DataFrame 객체 생성
df.columns = ['No1', 'No2', 'No3', 'No4'] #? 열 레이블 정의
df.index = pd.date_range('2000-1-1', periods=9, freq='M') #? 인덱스를 시계열로 정의

# TODO: 데이터 분석
df.info() #? 데이터, 열, 인덱스에 대한 메타정보 반환
df.describe() #? 열에 대한 기본적인 통계 정보 반환

df.sum() #? 열의 합 계산
df.mean() #? 열의 평군 계산
df.std()  #? 열의 표준편차 계산
df.cumsum() #? 열의 누적 합계 계산

# TODO: GroupBy 연산
df['Quarter'] = ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q3', 'Q3', 'Q3']
#* 데이터 그룹화를 위해 각 데이터에 분기 정보 추가

groups = df.groupby('Quarter') #? 분기 열을 기준으로 그룹화
groups.size() #? 각 그룹의 크기
groups.mean() #? 각 그룹의 평균
groups.max() #? 각 그룹의 최댓값
groups.min() #? 각 그룹의 최솟값

# TODO: 병합, 조인, 머지
df1 = pd.DataFrame(['100', '200', '300', '400'], columns=['A',], index=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(['10', '20', '30'], columns=['B',], index=['e', 'b', 'd'])

pd.concat((df1, df2)) #? 병합
df1.join(df2, how='left') #? 좌측 조인
df1.join(df2, how='right') #? 우측 조인
df1.join(df2, how='inner') #? 내부 조인
df1.join(df2, how='outer') #? 외부 조인

c = pd.DataFrame([250, 150, 50], index=['b', 'c', 'd'])
df1['C'] = c
df2['C'] = c

pd.merge(df1, df2) #? 머지