import numpy as np
from Utils.get_year_deltas import get_year_deltas

class ConstantShortRate():
  """고정 단기 이자율을 사용하여 할인율을 계산하는 클래스

  Attributes:
    name (str): 객체의 이름
    short_rate (float): 고정 이자율

  Method:
    get_discount_factors: 주진 날짜 리스트에 대한 할인율 계산
  """
  def __init__(self, name:str, short_rate:float):
    """고정 단기 이자 할인을 위한 클래스

    Args:
      name (str): 객체의 이름
      short_rate (float): 고정 이자율
    """
    self.name = name
    self.short_rate = short_rate

  def get_discount_factors(self, date_list:list, isDatetime:bool=True) -> np.ndarray:
    """주어진 날짜 리스트에 대해 할인율 계산

    Args:
      data_list (list): 날짜 리스트
      isDatetime (bool, optional): datetime 객체 리스트 여부 (기본값은 True)

    Returns:
      np.ndarray: 할인율이 계산된 배열
    """
    if isDatetime is True:
      dlist = get_year_deltas(date_list) #* 날짜 리스트를 연수로 변환
    else:
      dlist = np.array(date_list)

    dflist = np.exp(-self.short_rate * dlist) #* 할인율 계산
    return np.array((date_list, dflist)).T
