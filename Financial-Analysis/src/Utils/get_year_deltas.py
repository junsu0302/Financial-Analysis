import numpy as np

def get_year_deltas(date_list:list, day_count:float=365.) -> np.ndarray:
  """날자 벡터를 연수로 환산 (초기 날짜는 0)

  주어진 날짜 리스트의 시작 날짜를 기준으로 연수 변환한다.
  이는 금융 분석에서 다양한 시간 기반 계산에 유용하다.

  Args:
    date_list (list): datetime 객체 모음
    day_count (float, optional): 1년의 날짜 수 (기본값은 365.)

  Returns:
    numpy.ndarray: 각 날짜를 입력 배열의 첫 날짜로부터 연수로 변환한 numpy 배열 (소수 2째자리)
  """
  start = date_list[0] #* 기준이 되는 시작 날짜
  delta_list = [(date - start).days / day_count for date in date_list] #* 각 날짜를 기준 날짜로부터 연수로 변환
  return np.round(np.array(delta_list), 2) #* 소수 둘쨰자리까지 (반올림)