from typing import Union
import numpy as np
from Valuation.valuation import Valuation

class MCSEuropean(Valuation):
  """
  Monte Carlo Simulation을 이용한 유럽형 옵션 가치 평가 클래스

  Methods:
    generate_payoff(fixed_seed): 옵션의 페이오프 생성
    present_value(accuracy, fixed_seed, full): 옵션의 현재 가치 계산
  """
  def generate_payoff(self, fixed_seed:bool=False) -> np.ndarray:
    """
    옵션의 페이오프를 생성
    payoff : 옵션이 만기될 떄 기초 자산의 가격에 따라 옵션 보유자가 받게 되는 금액
    옵션의 페이오프는 가초 자산 가격과 옵션의 종류(call or put)에 따라 결정

    Args:
      fixed_seed (bool, optional): 고정된 시드 사용 여부 (기본값은 False)

    Returns:
      numpy.ndarray: 옵션 페이오프
    """
    try:
      strike = self.strike #* 옵션 행사가 설정
    except:
      pass
    
    paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed) #* 시뮬레이션된 자산 가격 경로를 가져옴
    time_grid = self.underlying.time_grid #* 시뮬레이션 시간 그리드를 가져옴
    
    try:
      time_index = np.where(time_grid == self.maturity)[0] #* 만기 날짜
      time_index = int(time_index)
    except:
      print('Maturity date not in time grid of underlying')

    maturity_value = paths[time_index] #* 만기 시점의 자산 가격 경로
    mean_value = np.mean(paths[:time_index], axis=1) #* 만기 이전까지의 자산 가격 평균
    max_value = np.amax(paths[:time_index], axis=1)[-1] #* 만기 이전까지의 자산 가격 중 최대값
    min_value = np.amin(paths[:time_index], axis=1)[-1] #* 만기 이전까지의 자산 가격 중 최소값
    
    try:
      payoff = eval(self.payoff_func) #* 주어진 페이오프 함수를 평가
      return payoff
    except:
      print('Error evaluating payoff function')  #* 페이오프 함수 평가 중 에러 발생 시 메시지 출력


  def present_value(self, accuracy:int=6, fixed_seed:bool=False, full:bool=False) -> Union[float, tuple]:
    """
    옵션의 현재 가치를 계산

    Args:
      accuracy (int, optional): 결과의 소수점 자릿수 (기본값은 6)
      fixed_seed (bool, optional): 고정된 시드 사용 여부 (기본값은 False)
      full (bool, optional): 전체 결과 반환 여부 (기본값은 False)

    Returns:
      float or tuple: 옵션의 현재 가치 (full이 True인 경우 할인된 캐시플로우 포함)
    """
    cash_flow = self.generate_payoff(fixed_seed=fixed_seed) #* 페이오프를 생성
    discount_factor = self.discount_curve.get_discount_factors((self.pricing_date, self.maturity))[0, 1] #* 할인 인수 계산
    result = discount_factor * np.sum(cash_flow) / len(cash_flow) #* 현재 가치 계산

    if full:
      return round(result, accuracy), discount_factor * cash_flow #* 전체 결과 반환
    else:
      return round(result, accuracy) #* 현재 가치만 반환
