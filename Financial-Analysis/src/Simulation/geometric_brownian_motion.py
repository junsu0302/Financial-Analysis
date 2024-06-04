import numpy as np
import datetime
from Valuation.market_environment import MarketEnvironment
from Simulation.sn_random_numbers import sn_random_numbers
from Simulation.simulation import Simulation

class GeometricBrownianMotion(Simulation):
  """기하 브라운 운동(GMB)을 이용한 자산 가격 시뮬레이션 클래스

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
  """
  def __init__(self, name:str, market_env:MarketEnvironment, corr:bool=False):
    """기하 브라운 운동(GMB)을 이용한 자산 가격 시뮬레이션

    Args:
      name (str): 시뮬레이션 이름
      market_env (MarketEnvironment): 시장 환경 객체
      corr (bool, optional): 상관관계 여부 (기본값은 False)
    """
    super(GeometricBrownianMotion, self).__init__(name, market_env, corr)

  def update(self, initial_value:float=None, volatility:float=None, final_date:datetime=None):
    """시뮬레이션 속성 업데이트

    Args:
      initial_value (float, optional): 초기 자산 가격
      volatility (float, optional): 자산 가격 변동성
      final_date (datetime, optional): 시뮬레이션 종료 날짜
    """
    if initial_value is not None:
      self.initial_value = initial_value #* 초기 자산 가격 업데이트
    
    if volatility is not None:
      self.volatility = volatility #* 변동성 업데이트

    if final_date is not None:
      self.final_date = final_date #* 종료 날짜 업데이트
    
    self.instrument_values = None #* 자산 가격 경로 초기화

  def generate_paths(self, fixed_seed:bool=False, day_count:float=365.):
    """기하 브라운 운동 경로 생성

    Args:
      fixed_seed (bool, optional): 고정된 시드 사용 여부 (기본값은 False)
      day_count (float, optional): 1년의 날짜 수 (기본값은 365)
    """
    if self.time_grid is None:
      self.generate_time_grid() #* 시간 그리드 생성

    M = len(self.time_grid) #* 시간 그리드의 길이
    I = self.paths #* 시뮬레이션 경로 수
    paths = np.zeros((M, I)) #* 경로 배열 초기화
    paths[0] = self.initial_value #* 초기 자산 가격 설정

    if not self.correlated:
      rand = sn_random_numbers((1, M, I), fixed_seed=fixed_seed) #* 난수 생성
    else:
      rand = self.random_numbers #* 상관관계가 있는 경우 기존 난수 사용

    short_rate = self.discount_curve.short_rate #* 단기 이자율 설정

    for t in range(1, len(self.time_grid)):
      if not self.correlated:
        ran = rand[t] #* 난수 설정
      else:
        ran = np.dot(self.cholesky_matrix, rand[:, t, :]) #* 상관관계가 있는 경우 코레스키 분해 적용
        ran = ran[self.rn_set] #* 난수 세트 적용

      dt = (self.time_grid[t] - self.time_grid[t-1]).days / day_count #* 시간 간격 계산

      paths[t] = paths[t-1] * np.exp((short_rate - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * ran) #* 기하 브라운 운동 경로 생성
    self.instrument_values = paths 