import numpy as np
import sympy as sy
import scipy.integrate as sci

def f(x):
  return np.sin(x) + 0.5 * x

a, b = 0.5, 9.5

# TODO: 수치 적분

# sci.fixed_quad(f, a, b)[0] #? 가우스 구적법
# sci.quad(f, a, b)[0] #? 적응 구적법
# sci.romberg(f, a, b) #? 롬베르크 적분법

# TODO: 심볼릭 연산
#? 심볼 정의
x = sy.Symbol('x')
y = sy.Symbol('y')

sy.solve(x ** 2 + y ** 2) #? 방성식 연산(해당 방정식의 모든 해 반환)

a, b = sy.symbols('a b') #? 적분 구간 정의
I = sy.Integral(sy.sin(x) + 0.5 * x, (x, a, b)) #? 적분 객체 정의
int_func = sy.integrate(sy.sin(x) + 0.5 * x, x) #? 부정적분 유도
int_func_limits = sy.integrate(sy.sin(x) + 0.5 * x, (x, a, b)) #? 적분 연산
int_func_limits.subs({a: 0.5, b: 9.5}).evalf() #? 구간의 수치적 적분 연산
