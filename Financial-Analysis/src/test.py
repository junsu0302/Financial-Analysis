import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from Valuation.get_year_deltas import get_year_deltas
from Valuation.constant_short_rate import ConstantShortRate

dates = [dt.datetime(2020, 1, 1), dt.datetime(2020, 7, 1), dt.datetime(2021, 1, 1)]

csr = ConstantShortRate('csr', 0.05)

deltas = get_year_deltas(dates)

print()
print('dates')
for idx in dates:
  print(idx)
print()

print('csr.get_discount_factors (dates)')
print(csr.get_discount_factors(dates))
print()

print('deltas')
print(deltas)
print()

print('get_discount_factors (deltas)')
print(csr.get_discount_factors(deltas, isDatetime=False))