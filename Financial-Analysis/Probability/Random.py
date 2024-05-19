import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

npr.seed(100) #? 시드값 고정
np.set_printoptions(precision=4) #? 출력 자릿수 결정

# TODO: 난수 생성
""" 
#? numpy.random의 난수 생성 함수
#* rand            | 주어진 형태로 난수 생성
#* randn           | 표준정규분포의 난수 생성
#* randint         | [low, high) 범위의 난수 생성
#* random_integers | [low, hith] 범위의 난수 생성
#* random_sample   | [0.0, 1.0) 범위의 float 난수 생성
#* random          | [0.0, 1.0) 범위의 float 난수 생성
#* randf           | [0.0, 1.0) 범위의 float 난수 생성
#* sample          | [0.0, 1.0) 범위의 float 난수 생성
#* choice          | 주어진 1차원 배열에서 무작위로 샘플 추출
#* bytes           | 바이트 난수 생성
"""

# sample_size = 500
# rn1 = npr.rand(sample_size, 3) #? 균일 분포 난수
# rn2 = npr.randint(0, 10, sample_size) #? 주어진 ㄱ구간에 대한 난수
# rn3 = npr.sample(size=sample_size) #? 균일 분포 난수
# a = [0, 25, 50, 75, 100]
# rn4 = npr.choice(a, size=sample_size) #? 리스트에서 무작워로 선택된 값

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# ax1.hist(rn1, bins=25, stacked=True)
# ax1.set_title('rand')
# ax1.set_ylabel('frequency')
# ax2.hist(rn2, bins=25)
# ax2.set_title('randint')
# ax3.hist(rn3, bins=25)
# ax3.set_title('sample')
# ax3.set_ylabel('frequency')
# ax4.hist(rn4, bins=25)
# ax4.set_title('choice')

# TODO: 확률 분포 법칙을 따르는 난수 생성
""" 
#? numpy.random의 난수 생성 함수
#* beta                 | [0, 1]의 베타 분포 - 베이지안 통계 및 신뢰 구간 계산
#* binomial             | 이항 분포 - 이진 결과 실험
#* chisquare            | 카이 제곱 분포 - 독립성, 적합도, 통계적 가설 검증
#* dirichlet            | 디리클레 분포 - 베이지안 통계 및 토픽 모델링
#* exponential          | 지수 분포 - 사건 간의 시간 간격 모델링
#* f                    | F 분포 - 두 표본의 분산 비교
#* gamma                | 감마 분포 - 대기 시간 모델링, 신뢰성 분석
#* geometric            | 기하 분포 - 첫 성공까지의 실패 횟수 모델링
#* gumbel               | 검벨 분포 - 최대값, 최소값의 분포 모델링
#* hypergeometric       | 초기하 분포 - 비복원 추출에서 성공 횟수 모델링
#* laplace              | 라플라스 분포 (이중지수분포) - 신호 처리 및 금융 모델링
#* logistic             | 로지스틱 분포 - S-곡선 형태 데이터 모델링
#* lognormal            | 로그 정규 분포 - 지수적으로 분포된 데이터(주가, 소득 분포 등) 모델링
#* logseries            | 로그 계열 분포 - 회귀 사건의 빈도수 모델링
#* multinomial          | 다항 분포 - 여러 범주에서 발생하는 횟수 모델링
#* multivariate_normal  | 다변량 정규 분포 - 여러 변수의 공통 분포 모델링
#* negative_binomial    | 음이항 분포 - 성공 횟수에 도달하기까지의 실패 횟수 모델링
#* noncentral_chisquare | 비중심 카이 제곱 분포 - 카이 제곱 검정에서 비중심 매개변수를 포함
#* noncentral_f         | 비중심 F 분포 - F 검정에서 비중심 매개변수 포함
#* normal               | 가우시안 정규 분포 - 측정 오류 모델링
#* pareto               | 제 2종 파레토 분포 - 80-20 법칙에 관련된 데이터 모델링
#* poisson              | 포아송 분포 - 단위 시간 or 단위 공간에서 사건 발생 횟수 모델링
#* power                | 양의 지수 a-1을 갖는 파워 분포 - 특정한 분포 형태를 갖는 데이터 모델링
#* rayleigh             | 레일리 분포 - 신호 강도 등에서 사용
#* standard_cauchy      | 코시 분포 - 중심 극한 정리의 대안으로 사용
#* standard_exponential | 표준지수분포 - 대기 시간 모델링
#* standard_gamma       | 표준감마분포 - 신뢰성 분석 및 대기 시간 모델링
#* standard_normal      | 표준정규분포 (평균 = 0, 표준편차 = 1) - 다양한 통계 분석 모델링에 사용
#* standard_t           | 표준 스튜던트 t 분포 - 작은 표본 크기의 데이터 분석
#* triangular           | 삼각 분포 - 주어진 최소값, 최빈값, 최대값을 갖는 데이터 모델링
#* uniform              | 균일 분포 - 주어진 범위 내에서 균일하게 분포된 데이터 모델링
#* vonmises             | 폰 미제스 분포 - 주기적 데이터 모델링
#* wald                 | 왈드 분포 (역 가우시안 분포) - 신뢰성 분석
#* weibull              | 베이불 분포 - 신뢰성 분석
#* zipf                 | 지프 분석 - 빈도, 분포 문제에 사용
"""

sample_size = 500

# 16개 분포의 난수 생성
rn1 = np.random.normal(0, 1, sample_size)  # 정규 분포
rn2 = np.random.lognormal(0, 1, sample_size)  # 로그 정규 분포
rn3 = np.random.exponential(1, sample_size)  # 지수 분포
rn4 = np.random.poisson(5, sample_size)  # 포아송 분포
rn5 = np.random.binomial(10, 0.5, sample_size)  # 이항 분포
rn6 = np.random.geometric(0.35, sample_size)  # 기하 분포
rn7 = np.random.negative_binomial(10, 0.5, sample_size)  # 음이항 분포
rn8 = np.random.chisquare(2, sample_size)  # 카이 제곱 분포
rn9 = np.random.gamma(2, 1, sample_size)  # 감마 분포
rn10 = np.random.beta(0.5, 0.5, sample_size)  # 베타 분포
rn11 = np.random.uniform(0, 1, sample_size)  # 균일 분포
rn12 = np.random.multinomial(20, [1/4.0]*4, size=sample_size).flatten()  # 다항 분포
rn13 = np.random.f(2, 5, sample_size)  # F 분포
rn14 = np.random.standard_normal(sample_size)  # 표준 정규 분포
rn15 = np.random.standard_t(10, sample_size)  # 표준 t 분포
rn16 = np.random.weibull(1.5, sample_size)  # 와이블 분포

fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 12))

#? 정규 분포 - 주가 수익률, 금융 자산 수익률의 변동성
#* 중앙 극한 정리에 의해 많은 데이터가 정규 분포를 따름
ax = axs[0, 0]
ax.hist(rn1, bins=25)
ax.set_title('Normal')

#? 로그 정규 분포 - 주가, 자산 가격 모형화
#* 시간에 따라 변하는 데이터 분석
ax = axs[0, 1]
ax.hist(rn2, bins=25)
ax.set_title('Lognormal')

#? 지수 분포 - 주가 변동 사이의 시간, 도산 시간 모델링
#* 사건 사이의 시간 간격 모델링
ax = axs[0, 2]
ax.hist(rn3, bins=25)
ax.set_title('Exponential')

#? 포아송 분포 - 주가 변동 횟수, 도산 사건의 수 모델링
#* 단위 시간 내에 발생하는 사건의 횟수 모델링
ax = axs[0, 3]
ax.hist(rn4, bins=range(rn4.min(), rn4.max()+2), align='left')
ax.set_title('Poisson')

#? 이항 분포 - 옵견 가격 결정 모형, 성공/실패 모델링
#* 고정된 수의 독립 시도에서 성공 횟수 모델링
ax = axs[1, 0]
ax.hist(rn5, bins=range(rn5.min(), rn5.max()+2), align='left')
ax.set_title('Binomial')

#? 기하 분포 - 첫 성공까지 필요한 시도 수 모델링
#* 첫 성공까지 걸리는 시도의 횟수 모델링
ax = axs[1, 1]
ax.hist(rn6, bins=range(rn6.min(), rn6.max()+2), align='left')
ax.set_title('Geometric')

#? 음이항 분포 - 성공에 달성하기까지의 실패 횟수 모델링
#* 성공에 달성하기까지의 실패 횟수 모델링
ax = axs[1, 2]
ax.hist(rn7, bins=range(rn7.min(), rn7.max()+2), align='left')
ax.set_title('Negative Binomial')

#? 카이 제곱 분포 - 분산 분석, 주가 변동성 테스트
#* 표본 분산과 이론적 분산을 비교하는 테스트
ax = axs[1, 3]
ax.hist(rn8, bins=25)
ax.set_title('Chi-Square')

#? 감마 분포 - 대기 시간, 청구액 모델링
#* 지수 분포의 일반화
ax = axs[2, 0]
ax.hist(rn9, bins=25)
ax.set_title('Gamma')

#? 베타 분포 - 비율 데이터 모델링, 주식의 베타 값 분석
#* 0~1 사이의 값으로 제한된 확률 변수 분포 모델링
ax = axs[2, 1]
ax.hist(rn10, bins=25)
ax.set_title('Beta')

#? 균일 분포 - 난수 생성, 몬테카를로 시뮬레이션
#* 동일한 확률로 발생하는 사건 모델링
ax = axs[2, 2]
ax.hist(rn11, bins=25)
ax.set_title('Uniform')

#? 다항 분포 - 포트폴리오 선택
#* 여러 범주의 확률을 모델링
ax = axs[2, 3]
counts = np.sum(rn12.reshape(-1, 4), axis=0)
ax.bar(range(len(counts)), counts)
ax.set_title('Multinomial')

#? F 분포 - 두 모집단의 분산 비교, 회귀 분석
#* 두 카이 제곱 분포 간의 비율 모델링
ax = axs[3, 0]
ax.hist(rn13, bins=25)
ax.set_title('F')

#? 표준 정규 분포 - 표준화된 데이터를 통한 분석
#* 중앙 극한 정리에 의해 많은 데이터가 정규 분포를 따름
ax = axs[3, 1]
ax.hist(rn14, bins=25)
ax.set_title('Standard Normal')

#? 표준 t 분포 - 작은 표본에서의 평균 비교, 수익률 분석
#* 표본 크기가 작을 때의 평균 테스트
ax = axs[3, 2]
ax.hist(rn15, bins=25)
ax.set_title('Standard T')

#? 와이블 분포 - 생존 분석, 신용 리스크 모델링
#* 고장 시간 및 생존 분석
ax = axs[3, 3]
ax.hist(rn16, bins=25)
ax.set_title('Weibull')

plt.tight_layout()
plt.show()