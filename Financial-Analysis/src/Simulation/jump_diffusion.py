import datetime
import numpy as np
from Environment.market_environment import MarketEnvironment
from Simulation.simulation import Simulation
from Simulation.sn_random_numbers import sn_random_numbers

class JumpDiffusion(Simulation):
  """
  s점프-확산 모형을 이용한 자산 가격 시뮬레이션 클래스

  Attributes:
    lamb (float): 점프 강도 (단위 시간당 평균 점프 횟수)
    mu (float): 점프 크기의 평균 (점프 크기의 로그)
    delta (float): 점프 크기의 표준편차

  Methods:
    update: 시뮬레이션 속성 업데이트
    generate_paths: 시뮬레이션 경로 생성
  """
  def __init__(self, name:str, market_env:MarketEnvironment, corr:bool=False):
    """
    점프-확산 모형을 이용한 자산 가격 시뮬레이션

    Args:
      name (str): 시뮬레이션 이름
      market_env (MarketEnvironment): 시장 환경 객체
      corr (bool, optional): 상관관계 여부 (기본값은 False)
    """
    super(JumpDiffusion, self).__init__(name, market_env, corr)
    self.lamb = market_env.get_constant('lambda') #* 점프 강도 설정
    self.mu = market_env.get_constant('mu') #* 점프 크기의 평균 설정
    self.delta = market_env.get_constant('delta') #* 점프 크기의 표준편차 설정

  def update(self, initial_value:float=None, volatility:float=None, lamb:float=None, mu:float=None, delta:float=None, final_date:datetime=None):
    """
    시뮬레이션 속성 업데이트

    Args:
      initial_value (float, optional): 초기 자산 가격
      volatility (float, optional): 자산 가격 변동성
      lamb (float, optional): 점프 강도
      mu (float, optional): 점프 크기의 평균
      delta (float, optional): 점프 크기의 표준편차
      final_date (datetime, optional): 시뮬레이션 종료 날짜
    """
    if initial_value is not None:
      self.initial_value = initial_value #* 초기 자산 가격 업데이트
    if volatility is not None:
      self.volatility = volatility #* 변동성 업데이트
    if lamb is not None:
      self.lamb = lamb #* 점프 강도 업데이트
    if mu is not None:
      self.mu = mu #* 점프 크기의 평균 업데이트
    if delta is not None:
      self.delta = delta #* 점프 크기의 표준편차 업데이트
    if final_date is not None:
      self.final_date = final_date #* 종료 날짜 업데이트
    self.instrument_values = None #* 기존 경로 초기화

  def generate_paths(self, fixed_seed:bool=False, day_count:float=365.):
    """
    점프-확산 모형을 이용하여 시뮬레이션 경로를 생성

    Args:
      fixed_seed (bool, optional): 랜덤 시드 고정 여부 (기본값은 False)
      day_count (float, optional): 1년을 몇 일로 계산할 것인지 (기본값은 365)
    """
    if self.time_grid is None:
      self.generate_time_grid() #* 타임 그리드 생성

    M = len(self.time_grid) #* 시뮬레이션 기간 내의 시간 단계 수
    I = self.paths #* 시뮬레이션 경로 수
    paths = np.zeros((M, I)) #* 경로 배열 초기화
    paths[0] = self.initial_value #* 초기 자산 가격 설정

    if self.correlated is False:
      sn1 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed) #* 상관관계가 없는 경우 난수 생성
    else:
      sn1 = self.random_numbers #* 상관관계가 있는 경우 사전에 생성된 난수 사용
      
    sn2 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed) #* 점프 크기를 위한 난수 생성
    rj = self.lamb * (np.exp(self.mu + 0.5 * self.delta ** 2) - 1) #* 점프 위험 중립 조정
    
    short_rate = self.discount_curve.short_rate #* 단기 이자율 설정

    for t in range(1, len(self.time_grid)):
      if self.correlated is False:
        ran = sn1[t] #* 난수 설정
      else:
        ran = np.dot(self.cholesky_matrix, sn1[:, t, :]) #* 상관관계가 있는 경우 코레스키 분해 적용
        ran = ran[self.rn_set] #* 난수 세트 적용
      dt = (self.time_grid[t] - self.time_grid[t-1]).days / day_count #* 시간 간격 계산
      poi = np.random.poisson(self.lamb * dt, I) #* 포아송 분포에 따른 점프 수 계산
      paths[t] = paths[t-1] * (np.exp((short_rate - rj - 
                                       0.5 * self.volatility ** 2) * dt +
                                       self.volatility * np.sqrt(dt) * ran) +
                                       (np.exp(self.mu + self.delta * sn2[t]) - 1) * poi) #* 경로 생성
      self.instrument_values = paths
