import numpy as np

def sn_random_numbers(shape:tuple, antithetic:bool=True, moment_matching:bool=True, fixed_seed:bool=False) -> np.ndarray:
  """표준 정규 분포 난수를 생성

  Args:
    shape (tuple): 생성할 배열의 형태 (o, n, m)
    antithetic (bool, optional): True면 분산 감소를 위해 대칭 변수를 생성 (기본값은 True)
    moment_matching (bool, optional): True면 난수를 평균 0, 표준편차 1로 조정 (기본값은 True)
    fixed_seed (bool, optional): True면 재현성을 위해 고정된 시드 설정 (기본값은 False)

  Returns:
    numpy.ndarray: 지정된 형태로 생성된 난수 배열 (o, n, m)
  """
  #? 시드 고정
  if fixed_seed:
    np.random.seed(1000) #* 고정 시드
  
  #? 분산 감소를 위해 대칭 변수 생성
  if antithetic:
    ran = np.random.standard_normal((shape[0], shape[1], shape[2] // 2)) #* 난수를 절반만 생성
    ran = np.concatenate((ran, -ran), axis=2) #* 반대 부호의 난수를 결합하여 대칭적인 분포 생성
  else:
    ran = np.random.standard_normal(shape) #* 일반적인 난수 생성
  
  #? 난수를 평균 0, 표준펀차 1로 조정
  if moment_matching:
    ran = ran - np.mean(ran) #* 난수의 평균을 0으로 조정
    ran = ran / np.std(ran) #* 난수의 표준편차를 1로 조정
  
  if shape[0] == 1:
    return ran[0]
  else:
    return ran