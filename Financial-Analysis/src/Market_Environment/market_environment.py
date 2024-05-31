import datetime

class MarketEnvironment():
  """가치 평가에 필요한 시장 환경을 모델링하는 클래스

  Attributes:
    name (str): 시장 환경의 이름
    pricing_date (datetime): 평가 기준 날짜
    constants (dict): 상수를 저장하는 딕셔너리
    lists (dict): 리스트를 저장하는 딕셔너리
    curves (dict): 시장 커브를 저장하는 딕셔너리

  Methods:
    add_constant(key, constant) : 상수(모델 인수 등) 추가
    get_constant(key) : 상수(모델 인수 등) 반환
    add_list(key, list) : 리스트(기초상품 등) 추가
    get_list(key) : 리스트(기초상품 등) 반환
    add_curve(key, curve) : 시장 커브(이자율 커브 등) 추가
    get_curve(key) : 시장 커브(이자율 커브 등) 반환
    add_environment(env): 다른 MarketEnvironment 객체의 데이터를 현재 객체에 병합
  """
  def __init__(self, name:str, pricing_date:datetime):
    """가치 평가에 필요한 시장 환경을 모델링하는 클래스

    Args:
      name (str): 시장 환경의 이름
      pricing_date (datetime): 평가 기준 날짜
    """
    self.name = name
    self.pricing_date = pricing_date
    self.constants = {}
    self.lists = {}
    self.curves = {}

  def add_constant(self, key:str, constant:any):
    """상수를 추가하는 메서드

    Args:
      key (str): 상수를 식별할 키
      constant (any): 추가할 상수
    """
    self.constants[key] = constant

  def get_constant(self, key:str):
    """상수를 반환하는 메서드

    Args:
      key (str): 반환할 상수를 식별할 키

    Returns:
      Any: 상수 값
    """
    return self.constants[key]
  
  def add_list(self, key:str, list:any):
    """리스트를 추가하는 메서드

    Args:
      key (str): 리스트를 식별할 키
      list (any): 추가할 리스트
    """
    self.lists[key] = list

  def get_list(self, key:str):
    """리스트를 반환하는 메서드

    Args:
      key (str): 반환할 리스트를 식별할 키

    Returns:
      any: 리스트 값
    """
    return self.lists[key]
  
  def add_curve(self, key:str, curve:any):
    """시장 커브를 추가하는 메서드

    Args:
      key (str): 시장 커브를 식별할 키
      curve (any): 추가할 시장 커브
    """
    self.curves[key] = curve

  def get_curve(self, key:str):
    """시장 커브를 반환하는 메서드

    Args:
      key (str): 반환할 시장 커브를 식별할 키

    Returns:
      Any: 시장 커브 값
    """
    return self.curves[key]
  
  def add_environment(self, env):
    """다른 MarketEnvironment 객체의 데이터를 현재 객체에 병합하는 메서드

    Args:
      env (MarketEnvironment): 병합할 다른 MarketEnvironment 객체
    """
    self.constants.update(env.contants)
    self.lists.update(env.lists)
    self.curves.update(env.curves)