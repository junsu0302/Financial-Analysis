from Simulation.simulation import Simulation
from Environment.market_environment import MarketEnvironment

class DerivativesPosition():
  """
  파생상품 포지션 모델링 클래스

  Args:
    name (str): 객체 이름
    quantity (float): 포지션을 구성하는 자산/파생상품의 개수
    underlying (str): 파생상품의 자산/위험 요인 이름
    market_env (MarketEnvironment): 시장 환경 객체
    otype (str): 사용할 가치 평가 옵션
    payoff_func (str): 파생상품의 페이오프 연삭식
    
  Method:
    get_info(): 파생상품 포지션 정보 출력
  """
  def __init__(self, name:str, quantity:float, underlying:str, market_env:MarketEnvironment, otype:str, payoff_func:str):
    self.name = name
    self.quantity = quantity
    self.underlying = underlying
    self.market_env = market_env
    self.otype = otype
    self.payoff_func = payoff_func

  def get_info(self):
    print('NAME')
    print(self.name, '\n')
    print('QUANTITY')
    print(self.quantity, '\n')
    print('UNDERLYING')
    print(self.underlying, '\n')
    print('MARKET ENVIRONMENT')
    print('\n**Constants**')
    for key, value in self.market_env.constants.items():
      print(key, value)
    print('\n**Lists**')
    for key, value in self.market_env.lists.items():
      print(key, value)
    print('\n**Curve**')
    for key, value in self.market_env.curves.items():
      print(key, value)
    print('\nOPTION TYPE')
    print(self.otype, '\n')
    print('PAYOFF FUNCTION')
    print(self.payoff_func, '\n')