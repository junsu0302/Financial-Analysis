import matplotlib.pyplot as plt

def plot_option_stats(s_list, p_list, d_list, v_list):
  #? 그림 크기 설정
  plt.figure(figsize=(10, 9))

  #? 옵션 현재 가치 (Present Value) 그래프
  sub1 = plt.subplot(311)
  plt.plot(s_list, p_list, 'ro', label='Present Value')
  plt.plot(s_list, p_list, 'b-', alpha=0.6)
  plt.title('Option Present Value')
  plt.ylabel('Present Value')
  plt.legend(loc='best')
  plt.grid(True)
  plt.setp(sub1.get_xticklabels(), visible=False)

  #? 델타 (Delta) 그래프
  sub2 = plt.subplot(312)
  plt.plot(s_list, d_list, 'go', label='Delta')
  plt.plot(s_list, d_list, 'b-', alpha=0.6)
  plt.title('Option Delta')
  plt.ylabel('Delta')
  plt.legend(loc='best')
  plt.ylim(min(d_list) - 0.1, max(d_list) + 0.1)
  plt.grid(True)
  plt.setp(sub2.get_xticklabels(), visible=False)

  #? 베가 (Vega) 그래프
  sub3 = plt.subplot(313)
  plt.plot(s_list, v_list, 'yo', label='Vega')
  plt.plot(s_list, v_list, 'b-', alpha=0.6)
  plt.title('Option Vega')
  plt.xlabel('Initial Value of Underlying')
  plt.ylabel('Vega')
  plt.legend(loc='best')
  plt.grid(True)

  #? 그래프 보여주기
  plt.tight_layout()
  plt.show()
