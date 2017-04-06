# -*- coding: utf-8 -*-

# CH06
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

fig = plt.gcf()          #獲得當前圖表的instance
fig.set_size_inches(10,15) #設定圖表的尺寸

# 將 'x' 設為 sympy 可處理的變數
var('x')
# 將 x 正規化成lambda函數 f
f = lambda x: exp(-x**2/2)

# 產生100個介於-4 ~ 4之間,固定間距的連續數
x = np.linspace(-5,5,100)
# 依序把 x內的值 丟到 f內 將回傳的值丟回 y
y = np.array([f(v) for v in x],dtype='float')

plt.grid(False)      #顯示圖表背景格線
plt.title('Gaussian Curve') #圖表標題
plt.xlabel('X')     #X軸標籤名
plt.ylabel('Y')     #Y軸標籤名
plt.plot(x,y,color='red')  #依 x,y 畫出灰色的線
plt.fill_between(x,y,0,color='#c0f0c0') #依 x,y 用綠色填滿到 x軸 之間的空間
plt.show()

# ch07

import math

mu_0, sigma_0 = 98,10
print "mu_0", mu_0
print "sigma_0", sigma_0

# 常態函數分布累積method
def normal_cdf(x, mu=0, sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

#逼近Z值method
def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find approximate inverse using binary search"""
    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z, low_p = -10.0, 0  # normal_cdf(-10) is (very close to) 0
    hi_z, hi_p = 10.0, 1  # normal_cdf(10)  is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2  # consider the midpoint
        mid_p = normal_cdf(mid_z)  # and the cdf's value there
        if mid_p < p:
            # midpoint is still too low, search above it
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            # midpoint is still too high, search below it
            hi_z, hi_p = mid_z, mid_p
        else:
            break

    return mid_z

#求Z值的method
def normal_lower_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

#第一個問題
# a = 10% = 0.90機率
print "normal_lower_bound(0.90, mu_0, sigma_0)", normal_lower_bound(0.90, mu_0, sigma_0)
print

#第一題第二個問題
print "normal_lower_bound(0.85, mu_0, sigma_0) = ", normal_lower_bound(0.85, mu_0, sigma_0)

#第一題第三個問題，求顯著性
#normal_probability_below = normal_cdf
print "normal_probability_belowx,mu,sigma) = ", normal_cdf(84.12,98, 10)
print ("")