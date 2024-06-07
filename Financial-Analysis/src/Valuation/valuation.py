from Simulation.simulation import Simulation
from Environment.market_environment import MarketEnvironment

class Valuation():
  """
  금융 상품 가치 평가를 위한 클래스

  Attributes:
    name (str): 평가 이름
    payoff_func (str): 지급 함수
    underlying (Simulation): 기초 자산 시뮬레이션 객체
    pricing_date (datetime): 평가일
    maturity (datetime): 만기일
    currency (str): 통화
    frequency (str): 시뮬레이션 빈도
    paths (int): 시뮬레이션 경로 수
    discount_curve (object): 할인 곡선 객체
    strike (float): 행사가 (옵션의 경우)

  Methods:
    update(initial_value, volatility, strike, maturity): 평가 속성 업데이트
    delta(interval, accuracy): 델타 계산
    vega(interval, accuracy): 베가 계산
    generate_payoff(): 옵션의 페이오프 생성 (구현 필요)
    present_value(): 옵션의 현재 가치 계산 (구현 필요)
  """
  def __init__(self, name:str, underlying:Simulation, market_env:MarketEnvironment, payoff_func:str=''):
    """금융 상품 가치 평가를 위한 클래스

    Args:
      name (str): 평가 이름
      underlying (Simulation): 기초 자산 시뮬레이션 객체
      market_env (MarketEnvironment): 시장 환경 객체
      payoff_func (str, optional): 지급 함수 (기본값은 '')
    """
    self.name = name  #* 평가 이름 설정
    self.payoff_func = payoff_func  #* 지급 함수 설정
    self.underlying = underlying  #* 기초 자산 시뮬레이션 객체 설정
    self.pricing_date = market_env.pricing_date  #* 시장 환경 객체에서 평가일을 가져옴
    self.maturity = market_env.get_constant('maturity')  #* 만기일을 설정
    self.currency = market_env.get_constant('currency')  #* 통화 설정
    self.frequency = underlying.frequency  #* 시뮬레이션 빈도를 기초 자산 시뮬레이션 객체에서 가져옴
    self.paths = underlying.paths  #* 시뮬레이션 경로 수를 기초 자산 시뮬레이션 객체에서 가져옴
    self.discount_curve = underlying.discount_curve  #* 할인 곡선을 기초 자산 시뮬레이션 객체에서 가져옴
    self.underlying.special_dates.extend([self.pricing_date, self.maturity])  #* 특별 날짜에 평가일과 만기일을 추가

    try:
      self.strike = market_env.get_constant('strike') #* 행사가를 설정 (옵션의 경우)
    except:
      pass

  def update(self, initial_value=None, volatility=None, strike=None, maturity=None):
    """평가 속성 업데이트

    Args:
      initial_value (float, optional): 초기 자산 가격
      volatility (float, optional): 자산 가격 변동성
      strike (float, optional): 행사가
      maturity (datetime, optional): 만기일
    """
    if initial_value is not None:
      self.underlying.update(initial_value=initial_value)  #* 초기 자산 가격 업데이트
    if volatility is not None:
      self.underlying.update(volatility=volatility)  #* 자산 가격 변동성 업데이트
    if strike is not None:
      self.strike = strike  #* 행사가 업데이트
    if maturity is not None:
      self.maturity = maturity  #* 만기일 업데이트
      if maturity not in self.underlying.time_grid:
        self.underlying.special_dates.append(maturity)  #* 만기일을 특별 날짜에 추가
        self.underlying.instrument_values = None  #* 자산 가격 경로 초기화

  def delta(self, interval=None, accuracy=4):
    """델타 계산 : 기초 자산 가격 변화에 대한 옵션 가격의 민감도 측정
    델타는 기초 자산 가격이 1단위 변할 때 옵션 가격이 얼마나 변하는지를 나타낸다.

    Args:
      interval (float, optional): 초기 자산 가격 변화량 (기본값은 None)
      accuracy (int, optional): 결과의 소수점 자릿수 (기본값은 4)

    Returns:
      float: 델타 값
    """
    if interval is None:
      interval = self.underlying.initial_value / 50.  #* 초기 자산 가격의 1/50을 interval로 설정
    
    #? 전향 차분 근사화 (V(S+h) - V(S)) / h
    value_left = self.present_value(fixed_seed=True) #* V(S)
    initial_del = self.underlying.initial_value + interval #* interval만큼 이동
    self.underlying.update(initial_value=initial_del) #* 업데이트
    value_right = self.present_value(fixed_seed=True)  #* V(S+h)
    self.underlying.update(initial_value=initial_del - interval) #* 원래 초기 자산으로 복구
    delta = (value_right - value_left) / interval #* 델타 계산

    #? 수치 오류 정정
    if delta < -1.0:
      return -1.0
    elif delta > 1.0:
      return 1.0
    else:
      return round(delta, accuracy)
    
  def vega(self, interval=0.01, accuracy=4):
    """베가 계산 : 변동성 변화에 대한 옵션 가격의 민감도 측정
    베가는 기초 자산의 변동성이 1% 변할 때 옵션 가격이 얼마나 변하는지를 나타낸다.

    Args:
      interval (float, optional): 변동성 변화량 (기본값은 0.01)
      accuracy (int, optional): 결과의 소수점 자릿수 (기본값은 4)

    Returns:
      float: 베가 값
    """
    if interval < self.underlying.volatility / 50.:
      interval = self.underlying.volatility / 50.  #* 변동성의 1/50을 최소 interval로 설정
    
    #? 전향 차분 근사화 (V(S+h) - V(S)) / h
    value_left = self.present_value(fixed_seed=True) #* V(S)
    vola_del = self.underlying.volatility + interval #* interval만큼 이동
    self.underlying.update(volatility=vola_del) #* 업데이트
    value_riget = self.present_value(fixed_seed=True) #* V(S+h)
    self.underlying.update(volatility=vola_del - interval) #* 원래 변동성으로 복구
    vega = (value_riget - value_left) / interval #* 베가 계산
    
    return round(vega, accuracy) 
  
  def generate_payoff():
    raise NotImplementedError("The method generate_payoff() must be implemented in subclasses")

  def present_value():
    raise NotImplementedError("The method present_value() must be implemented in subclasses")