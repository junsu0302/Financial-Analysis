import numpy as np
import pandas as pd
import datetime
from Environment.market_environment import MarketEnvironment

class Simulation():
  """
  자산 가격 시뮬레이션을 위한 기본 클래스

  Attributes:
    name (str): 시뮬레이션 이름
    pricing_date (datetime): 평가일
    initial_value (float): 초기 자산 가격
    volatility (float): 자산 가격 변동성
    final_date (datetime): 시뮬레이션 종료 날짜
    currency (str): 자산 가격 통화
    frequency (str): 시뮬레이션 빈도
    paths (int): 시뮬레이션 경로 수
    discount_curve (object): 할인 곡선 객체
    instrument_values (np.ndarray): 시뮬레이션된 자산 가격 경로
    correlated (bool): 상관관계 여부
    cholesky_matrix (np.ndarray): 코레스키 분해 행렬 (상관관계가 있는 경우)
    rn_set (np.ndarray): 무작위 수 세트 (상관관계가 있는 경우)
    random_numbers (np.ndarray): 무작위 수 (상관관계가 있는 경우)
    time_grid (np.ndarray): 시뮬레이션 시간 그리드
    special_dates (list): 시뮬레이션에 포함할 특별 날짜

  Methods:
    update(inital_value, volatility, final_date): 시뮬레이션 속성 업데이트 (구현 필요)
    generate_paths(fixed_seed, day_count): 시뮬레이션 경로 생성 (구현 필요)
    generate_time_grid(): 시뮬레이션 시간 그리드 생성
    get_instrument_values(fixed_seed): 시뮬레이션된 자산 가격 경로 반환
  """
  def __init__(self, name:str, market_env:MarketEnvironment, corr:bool):
    """자산 가격 시뮬레이션을 위한 기본 클래스

    Args:
      name (str): 시뮬레이션 이름
      market_env (MarketEnvironment): 시장 환경 객체
      corr (bool): 상관관계 여부
    """
    self.name = name  #* 시뮬레이션의 이름을 설정
    self.pricing_date = market_env.pricing_date  #* 시장 환경 객체에서 평가일을 가져옴
    self.initial_value = market_env.get_constant('initial_value')  #* 초기 자산 가격을 설정
    self.volatility = market_env.get_constant('volatility')  #* 자산 가격의 변동성을 설정
    self.final_date = market_env.get_constant('final_date')  #* 시뮬레이션 종료 날짜를 설정
    self.currency = market_env.get_constant('currency')  #* 자산 가격의 통화를 설정
    self.frequency = market_env.get_constant('frequency')  #* 시뮬레이션 빈도를 설정
    self.paths = market_env.get_constant('paths')  #* 시뮬레이션 경로 수를 설정
    self.discount_curve = market_env.get_curve('discount_curve')  #* 할인 곡선을 설정
    self.instrument_values = None  #* 초기 자산 가격 경로를 None으로 설정
    self.correlated = corr #* 상관관계 여부를 설정
    
    if corr:  #? 상관관계가 있는 경우
      self.cholesky_matrix = market_env.get_list('cholesky_matrix') #* 코레스키 분해 행렬을 설정
      self.rn_set = market_env.get_list('rn_set')[self.name] #* 무작위 수 세트를 설정
      self.random_numbers = market_env.get_list('random_numbers') #* 무작위 수를 설정

    try:
      self.time_grid = market_env.get_list('time_grid') #* 시간 그리드를 설정
    except:
      self.time_grid = None #* 시간 그리드가 없으면 None으로 설정
    
    try:
      self.special_dates = market_env.get_list('special_dates') #* 특별 날짜를 설정
    except:
      self.special_dates = [] #* 특별 날짜가 없으면 빈 리스트로 설정

  def update():
    """시뮬레이션 속성 업데이트 (구현 필요)"""
    raise NotImplementedError("The method update() must be implemented in subclasses")

  def generate_paths():
    """시뮬레이션 경로 생성 (구현 필요)"""
    raise NotImplementedError("The method generate_paths() must be implemented in subclasses")

  def generate_time_grid(self):
    """
    시뮬레이션 시간 그리드를 생성

    이 함수는 시뮬레이션의 시간 축을 생성. 시작 날짜와 종료 날짜 사이의 날짜를 주어진 빈도에 따라 생성.
    특별 날짜가 있는 경우 이를 포함하여 시간 그리드를 생성.
    """
    start = self.pricing_date #* 시뮬레이션 시작 날짜
    end = self.final_date #* 시뮬레이션 종료 날짜
    time_grid = pd.date_range(start=start, end=end, freq=self.frequency).to_pydatetime() #* 시작 날짜와 종료 날짜 사이의 날짜 생성
    time_grid = list(time_grid) 

    if start not in time_grid: #* 시작 날짜가 시간 그리드에 없다면 추가
      time_grid.insert(0, start) 

    if end not in time_grid: #* 종료 날짜가 시간 그리드에 없으면 추가
      time_grid.append(end) 

    if len(self.special_dates) > 0: #* 특별 날짜가 있으면 추가
      time_grid.extend(self.special_dates) 
      time_grid = list(set(time_grid))
      time_grid.sort()

    self.time_grid = np.array(time_grid)

  def get_instrument_values(self, fixed_seed:bool=True) -> np.ndarray:
    """
    시뮬레이션된 자산 가격 경로를 반환

    이 함수는 자산 가격 경로를 시뮬레이션하고 반환한다. 
    고정된 시드를 사용하여 시뮬레이션을 반복 가능하게 만들 수 있다.

    Args:
      fixed_seed (bool, optional): 고정된 시드 사용 여부 (기본값은 True)

    Returns:
      numpy.ndarray: 시뮬레이션된 자산 가격 경로
    """
    if self.instrument_values is None: #* 자산 경로가 없으면 생성
      self.generate_paths(fixed_seed=fixed_seed, day_count=365.)
    elif fixed_seed is False: #* 고정된 시드가 아니면 자산 가격 경로를 다시 생성
      self.generate_paths(fixed_seed=fixed_seed, day_count=365.)
    
    return self.instrument_values