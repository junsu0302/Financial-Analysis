import numpy as np

# TODO: 배열 생성
tmp_list = np.array([0, 0.5, 1.0, 1.5, 2.0])
""" 
#? numpy를 활용하여 문자열 리스트인 ndarray 객체 a 생성
#* type(a) : numpy.ndarray
#* dtype : np.float
"""

# TODO: numpy.ndarray 내장 메서드
tmp_list.sum() #? 모든 원소의 총합
tmp_list.std() #? 모든 원소의 표준편차
tmp_list.cumsum() #? 모든 원소의 누적합

# TODO: numpy.ndarray 메타 정보
tmp_list.size #? 원소의 개수
tmp_list.itemsize #? 원소 하나에 사용된 바이트 수
tmp_list.ndim #? 차원의 수
tmp_list.shape #? 객체의 형상
tmp_list.dtype #? 원소의 자료형
tmp_list.nbytes #? 사용된 메모리 총량

# TODO: 형태 변환
shape_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

shape_list.reshape((3,5))