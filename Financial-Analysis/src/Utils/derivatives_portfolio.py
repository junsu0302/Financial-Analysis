import numpy as np
import pandas as pd

from Simulation.geometric_brownian_motion import GeometricBrownianMotion
from Simulation.jump_diffusion import JumpDiffusion
from Simulation.square_root_diffusion import SquareRootDiffusion
from Simulation.sn_random_numbers import sn_random_numbers

from Valuation.mcs_american import MCSAmerican
from Valuation.mcs_european import MCSEuropean

from Environment.market_environment import MarketEnvironment

#? 모델 정의
models = {
  'gbm': GeometricBrownianMotion, #* 기하 브라운 운동 모형
  'jd': JumpDiffusion,            #* 점프-확산 모형
  'srd': SquareRootDiffusion      #* 제곱근 확산 모형
}

#? 옵션 타입 정의
otypes = {
  'American': MCSAmerican,        #* 미국식 옵션 가치평가
  'European': MCSEuropean         #* 유럽식 옵션 가치평가
}

class DerivativesPortfolio():
  """
  파생상품 포트폴리오 클래스

  Attributes:
    name (str): 포트폴리오 이름
    positions (dict): 포지션 딕셔너리
    valuation_env (MarketEnvironment): 평가 환경 객체
    assets (dict): 자산 딕셔너리
    correlations (list, optional): 상관관계 리스트 (기본값은 None)
    fixed_seed (bool, optional): 랜덤 시드 고정 여부 (기본값은 False)
    underlyings (set): 기초자산 집합
    time_grid (np.array): 시간 그리드
    underlying_objects (dict): 기초자산 객체 딕셔너리
    valuation_objects (dict): 평가 객체 딕셔너리
    special_dates (list): 특별 날짜 리스트

  Methods:
    get_positions: 포지션 정보 출력
    get_statistics: 통계 정보 출력
  """
  def __init__(self, name:str, positions:dict, valuation_env:MarketEnvironment, assets:dict, correlations:list=None, fixed_seed:bool=False):
    """
    파생상품 포트폴리오

    Args:
      name (str): 포트폴리오 이름
      positions (dict): 포지션 딕셔너리
      valuation_env (MarketEnvironment): 평가 환경 객체
      assets (dict): 자산 딕셔너리
      correlations (list, optional): 상관관계 리스트 (기본값은 None)
      fixed_seed (bool, optional): 랜덤 시드 고정 여부 (기본값은 False)
    """
    self.name = name #* 포트폴리오 이름
    self.positions = positions #* 포지션 딕셔너리
    self.valuation_env = valuation_env #* 평가 환경 객체
    self.assets = assets #* 자산 딕셔너리
    self.correlations = correlations #* 상관관계 리스트
    self.fixed_seed = fixed_seed #* 랜덤 시드 고정 여부
    self.underlyings = set() #* 기초자산 집합
    self.time_grid = None #* 시간 그리드
    self.underlying_objects = {} #* 기초자산 객체 딕셔너리
    self.valuation_objects = {} #* 평가 객체 딕셔너리
    self.special_dates = [] #* 특별 날짜 리스트

    #? 포지션별로 시작일과 종료일 설정
    for pos in self.positions:
     #* 포지션의 시작일과 종료일을 고려하여 평가 환경의 시작일과 종료일 설정
      self.valuation_env.constants['starting_date'] = min(self.valuation_env.constants['starting_date'], positions[pos].market_env.pricing_date)
      self.valuation_env.constants['final_date'] = max(self.valuation_env.constants['final_date'], positions[pos].market_env.constants['maturity'])
      self.underlyings.add(positions[pos].underlying) #* 기초자산 집합에 추가
    start = self.valuation_env.constants['starting_date'] #* 시작일 설정
    end = self.valuation_env.constants['final_date'] #* 종료일 설정

    #? 주어진 시작일과 종료일 범위에서 주어진 빈도로 날짜 시퀀스를 생성하여 시간 그리드를 생성
    time_grid = pd.date_range(start=start, end=end, freq=self.valuation_env.constants['frequency']).to_pydatetime()
    time_grid = list(time_grid) #* 시간 그리드를 리스트로 변환

    #? 모든 포지션의 만기일을 고려하여 시간 그리드에 특별 날짜를 추가
    for pos in self.positions:
      maturity_date = positions[pos].market_env.constants['maturity'] #* 포지션의 만기일
      if maturity_date not in time_grid:
        time_grid.insert(0, maturity_date) #* 만기일이 시간 그리드에 없으면 추가
        self.special_dates.append(maturity_date) #* 특별 날짜 리스트에도 추가

    #? 시작일과 종료일이 시간 그리드에 없으면 추가
    if start not in time_grid:
      time_grid.insert(0, start)
    if end not in time_grid:
      time_grid.append(end)

    #? 평가 환경에 시간 그리드 추가
    time_grid = list(set(time_grid)) #* 중복 제거 후 다시 리스트로 변환
    time_grid.sort() #* 날짜를 오름차순으로 정렬
    self.time_grid = np.array(time_grid) #* 시간 그리드를 넘파이 배열로 변환
    self.valuation_env.add_list('time_grid', self.time_grid) #* 평가 환경에 시간 그리드 추가

    if correlations is not None:
      ul_list = sorted(self.underlyings) #* 기초자산 집합을 정렬하여 ul_list에 저장
      correlation_matrix = np.zeros((len(ul_list), len(ul_list))) #* 상관관계 행렬 초기화
      np.fill_diagonal(correlation_matrix, 1.0) #* 대각선을 1로 채움 (자기 자신과의 상관관계는 항상 1)

      #? 상관관계를 기반으로 상관관계 행렬 업데이트
      correlation_matrix = pd.DataFrame(correlation_matrix, index=ul_list, columns=ul_list)  
      for i, j, corr, in correlations:
        corr = min(corr, 0.999999999999) #* 최대 상관관계 값이 1이 되도록 설정
        #* (i, j)와 (j, i) 위치에 상관관계 값 설정
        correlation_matrix.loc[i, j] = corr  
        correlation_matrix.loc[j, i] = corr

      cholesky_matrix = np.linalg.cholesky(np.array(correlation_matrix)) #* 촐레스키 분해 수행하여 상관관계 행렬의 제곱근 행렬 생성

      #? 기초자산 인덱스를 기초자산 이름에 매핑하는 딕셔너리 생성
      rn_set = {
        asset: ul_list.index(asset) 
        for asset in self.underlyings
      }

      #? 난수 생성 및 평가 환경에 추가
      random_numbers = sn_random_numbers((len(rn_set), len(self.time_grid), self.valuation_env.constants['paths']), fixed_seed=self.fixed_seed)
      self.valuation_env.add_list('cholesky_matrix', cholesky_matrix)
      self.valuation_env.add_list('random_numbers', random_numbers)
      self.valuation_env.add_list('rn_set', rn_set)

    #? 기초자산을 순회하면서 해당 자산의 시장 환경을 가져옴
    for asset in self.underlyings:
      market_env = self.assets[asset]
      market_env.add_environment(valuation_env) #* 평가 환경을 시장 환경에 추가
      model = models[market_env.constants['model']] #* 모델 선택
      if correlations is not None:
        #* 상관관계가 지정된 경우, 점프-확산 모형 사용
        self.underlying_objects[asset] = model(asset, market_env, corr=True)
      else:
        #* 상관관계가 지정되지 않은 경우, 일반 확산 모형 사용
        self.underlying_objects[asset] = model(asset, market_env, corr=False)


    for pos in positions:
      #* 포지션마다 해당하는 가치 평가 클래스 선택
      valuation_class = otypes[positions[pos].otype]
      #* 포지션의 시장 환경 가져오기
      market_env = positions[pos].market_env
      #* 시장 환경에 평가 환경 추가
      market_env.add_environment(self.valuation_env)
      #* 포지션의 가치 객체 생성 및 저장
      self.valuation_objects[pos] = valuation_class(name=positions[pos].name,
                                                    market_env=market_env,
                                                    underlying=self.underlying_objects[positions[pos].underlying],
                                                    payoff_func=positions[pos].payoff_func)

    
  def get_positions(self):
    """모든 포지션에 대한 정보를 출력"""
    for pos in self.positions:
      bar = '\n' + 50 * '-'
      print(bar)
      self.positions[pos].get_info()
      print(bar)

  def get_statistics(self, fixed_seed:bool=False):
    """포지션 통계를 반환

    Args:
      fixed_seed (bool, optional): 난수 시드를 고정할지 여부 (기본값은 False)

    Returns:
      pandas.DataFrame: 포지션 통계 정보가 포함된 DataFrame
    """
    result_list = [] #* 결과를 저장할 리스트 생성
    for pos, value in self.valuation_objects.items():
      p = self.positions[pos] #* 현재 포지션 가져오기
      pv = value.present_value(fixed_seed=fixed_seed) #* 현재 가치 계산
      result_list.append([p.name,                       #* 포지션 이름 추가
                          p.quantity,                   #* 포지션 수량 추가
                          pv,                           #* 현재 가치 추가
                          value.currency,               #* 통화 추가
                          pv * p.quantity,              #* 포지션 가치 추가
                          value.delta() * p.quantity,   #* 델타 추가
                          value.vega() * p.quantity])   #* 베가 추가
    #* 결과를 DataFrame으로 변환하여 반환
    result_df = pd.DataFrame(result_list, columns=['name', 'quant', 'value', 'curr', 'pos_value', 'pos_delta', 'pos_vega'])
    return result_df
