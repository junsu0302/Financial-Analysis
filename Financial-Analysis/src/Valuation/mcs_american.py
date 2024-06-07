import numpy as np
from Valuation.valuation import Valuation

class MCSAmerican(Valuation):
  """
  Monte Carlo Simulation을 이용한 미국식 옵션 가치 평가 클래스

  Methods:
    generate_payoff(fixed_seed): 옵션의 페이오프 생성
    present_value(accuracy, fixed_seed, full): 옵션의 현재 가치 계산
  """
  def generate_payoff(self, fixed_seed:bool=False) -> tuple:
    """
    옵션의 페이오프를 생성
    payoff : 옵션이 만기될 떄 기초 자산의 가격에 따라 옵션 보유자가 받게 되는 금액
    옵션의 페이오프는 가초 자산 가격과 옵션의 종류(call or put)에 따라 결정

    Args:
      fixed_seed (bool, optional): 고정된 시드 사용 여부 (기본값은 False)

    Returns:
      tuple: 옵션 페이오프
    """
    try:
      strike = self.strike #* 옵션의 행사가격
    except AttributeError:
      strike = None

    paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed) #* 기초 자산의 시뮬레이션 경로
    time_grid = self.underlying.time_grid #* 시뮬레이션 시간 그리드

    #* 시작 시간 인덱스 설정
    try:
      time_index_start = int(np.where(time_grid == self.pricing_date)[0][0])
    except IndexError:
      raise ValueError("Pricing date not in time grid")

    #* 종료 시간 인덱스 설정
    if self.maturity in time_grid:
      time_index_end = int(np.where(time_grid == self.maturity)[0][0])
    else:
      time_index_end = np.argmin(np.abs(time_grid - self.maturity))
      
    #* 시뮬레이션 경로에서 해당 기간의 자산 가격
    instrument_values = paths[time_index_start:time_index_end+1]

    #* 페이오프 함수 계산
    try:
      payoff = eval(self.payoff_func, {"np": np, "instrument_values": instrument_values, "strike": strike})
    except Exception as e:
      raise ValueError("Error evaluating payoff function: {}".format(e))

    return instrument_values, payoff, time_index_start, time_index_end

  def present_value(self, accuracy:int=6, fixed_seed:bool=False, bf:int=5, full:bool=False):
    """
    옵션의 현재 가치를 계산

    Args:
      accuracy (int, optional): 결과의 소수점 자릿수 (기본값은 6)
      fixed_seed (bool, optional): 고정된 시드 사용 여부 (기본값은 False)
      bf (int, optional): 회귀에 사용되는 평균 관찰값의 수 (기본값은 5)
      full (bool, optional): 전체 결과 반환 여부 (기본값은 False)

    Returns:
      float or tuple: 옵션의 현재 가치 (full이 True인 경우 할인된 캐시플로우 포함)
    """
    instrument_values, inner_values, time_index_start, time_index_end = self.generate_payoff(fixed_seed=fixed_seed) #* 페이오프 생성
    time_list = self.underlying.time_grid[time_index_start:time_index_end + 1] #* 시간 그리드 설정
    discount_factors = self.discount_curve.get_discount_factors(time_list, isDatetime=True) #* 할인율 계산

    V = inner_values[-1] #* 마지막 시점에서의 페이오프

    #* 역순으로 최적의 옵션 가치 계산
    for t in range(len(time_list) - 2, 0, -1):
      df = discount_factors[t, 1] / discount_factors[t + 1, 1] #* 할인 인자 계산
      rg = np.polyfit(instrument_values[t], V * df, bf) #* 다항식 회귀
      C = np.polyval(rg, instrument_values[t]) #* 회귀 다항식 평가
      V = np.where(inner_values[t] > C, inner_values[t], V * df) #* 최적 가치 결정

    df = discount_factors[0, 1] / discount_factors[1, 1] #* 초기 할인 인자 계산
    result = df * np.sum(V) / len(V) #* 현재 가치 계산

    if full:
      return round(result, accuracy), df * V #* 전체 결과 반환
    else:
      return round(result, accuracy) #* 현재 가치만 반환
