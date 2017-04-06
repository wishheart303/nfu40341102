倩綾的筆記
=========================

以下是我使用Numpy、Pandas以及Seaborn所做出來的程式碼。(結合第2組和第3組所介紹的功能)。

其資料來源是http://amis.afa.gov.tw/fruit/FruitChartProdTransPriceVolumeTrend.aspx。

查詢台灣全部市場106年1月1日~3月30日的草莓的交易量(KG)。

日期 : 2017/03/30

作業
```python
# coding=UTF-8
import seaborn as sns
import pandas as pd
import numpy as np

## 以下分別為 x 和 y 的值，其形態為 list 。
Date = np.array(['03/01', '03/02', '03/03', '03/04', '03/05', '03/07', '03/08',
        '03/09', '03/10', '03/11', '03/12', '03/14', '03/15', '03/16',
        '03/17', '03/18', '03/19','03/21', '03/22', '03/23', '03/24',
        '03/25', '03/26', '03/28', '03/29', '03/30'])

 # x軸的值
Volumes = np.array([10934.5, 12922.5, 15602.5, 18457.5, 20035.0, 26097.5,  20104.0,
            16512.5, 15385.0, 16167.5, 16250.5, 25162.5, 18751.5, 17651.0,
           16036.0, 19933.5, 17460.0, 27273.0, 22006.0, 21021.0, 20627.0,
           25011.0, 22002.0, 26257.0, 15105.0, 14487.0])
 # y軸的值

x = np.sum(Volumes)
len_x = len(Volumes)
avg_volumes = x / len_x
print("平均交易量為: " + str(avg_volumes))



data_df = pd.DataFrame({"Date(Year = 106)": Date, ## 欄位名稱：值
                     "Volume": Volumes ## 欄位名稱：值
                         }
                       )

sns.factorplot(data = data_df, x="Date(Year = 106)", y="Volume", ci = None)
sns.plt.show() ## 顯示出來


```

日期 : 2017/04/06

課堂練習

CH06  Probability
```python
# coding=UTF-8
from __future__ import division
from collections import Counter
import math, random
from matplotlib import pyplot as plt

def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

def plot_normal_pdfs(plt):
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
    plt.plot(xs, [normal_pdf(x, sigma=1, mu=3) for x in xs], ':', label='mu=3,sigma=1')
    plt.plot(xs, [normal_pdf(x, mu=-4, sigma=0.5) for x in xs], '-.', label='mu=-1,sigma=1')
    plt.legend()
    plt.show()  
    
def random_ball():
    return random.choice(["B", "Y"])

a1 = 0
a2 = 0
a_both = 0
n = 1000000
random.seed(0)
for _ in range(n):
    get1 = random_ball()
    get2 = random_ball()
    if get1 == "B":
        a1 += 1
    if get1 == "B" and get2 == "B":
        a_both += 1
    if get2 == "B":
        a2 += 1


print "P(both):", a_both / n
print "P(get1): ", a1 / n
print "P(get2):", a2 / n
print "P(get1 , get2): ", a1 * a2 / n / n
print "P(get1 | get2) = P(both) / P(get2): ", (a_both / n) / (a2 / n)
print "P(get1 | get2) = P(get1 , get2) / P(get2) = P(get1) * P(get2) / P(get2) = P(get1): ", a1 / n

plot_normal_pdfs(plt)


```
課堂練習

CH07 Hypothesis and Inference
```python
from __future__ import division
from probability_all import normal_cdf, inverse_normal_cdf
import math, random

def normal_approximation_to_binomial(n, p):
    """finds mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

#####
#
# probabilities a normal lies in an interval
#
######
# CH06
# def normal_cdf(x, mu=0,sigma=1):
#     return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
#
#
# def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
#     """find approximate inverse using binary search"""
#
#     # if not standard, compute standard and rescale
#     if mu != 0 or sigma != 1:
#         return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
#
#     low_z, low_p = -10.0, 0  # normal_cdf(-10) is (very close to) 0
#     hi_z, hi_p = 10.0, 1  # normal_cdf(10)  is (very close to) 1
#     while hi_z - low_z > tolerance:
#         mid_z = (low_z + hi_z) / 2  # consider the midpoint
#         mid_p = normal_cdf(mid_z)  # and the cdf's value there
#         if mid_p < p:
#             # midpoint is still too low, search above it
#             low_z, low_p = mid_z, mid_p
#         elif mid_p > p:
#             # midpoint is still too high, search below it
#             hi_z, hi_p = mid_z, mid_p
#         else:
#             break
#
#     return mid_z


# the normal cdf _is_ the probability the variable is below a threshold
normal_probability_below = normal_cdf

# it's above the threshold if it's not below the threshold
def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo, mu, sigma)
    
# it's between if it's less than hi, but not less than lo
def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# it's outside if it's not between
def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1 - normal_probability_between(lo, hi, mu, sigma)

######
#
#  normal bounds
#
######


def normal_upper_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)
    
def normal_lower_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """returns the symmetric (about the mean) bounds 
    that contain the specified probability"""
    tail_probability = (1 - probability) / 2

    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        # if x is greater than the mean, the tail is above x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # if x is less than the mean, the tail is below x
        return 2 * normal_probability_below(x, mu, sigma)   

def count_extreme_values():
    extreme_value_count = 0
    for _ in range(100000):
        num_heads = sum(1 if random.random() < 0.5 else 0    # count # of heads
                        for _ in range(1000))                # in 1000 flips
        if num_heads >= 530 or num_heads <= 470:             # and count how often
            extreme_value_count += 1                         # the # is 'extreme'

    return extreme_value_count / 100000

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below    

##
#
# P-hacking
#
##

def run_experiment():
    """flip a fair coin 1000 times, True = heads, False = tails"""
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment):
    """using the 5% significance levels"""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531


##
#
# running an A/B test
#
##

def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

##
#
# Bayesian Inference
#
##

def B(alpha, beta):
    """a normalizing constant so that the total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:          # no weight outside of [0, 1]    
        return 0        
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)


if __name__ == "__main__":

    p = 0.99
    a = 0.46
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
    print "mu_0", mu_0
    print "sigma_0", sigma_0
    print "normal_two_sided_bounds("+ str(p) + ", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0)
    print

    p = 0.5
    a = 0.90
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
    print "mu_0", mu_0
    print "sigma_0", sigma_0
    print "normal_two_sided_bounds("+ str(p) + ", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0)
    print

```

日期 : 2017/04/06

作業

```python
# coding=UTF-8
import seaborn as sns
import pandas as pd
import numpy as np

## 以下分別為 x 和 y 的值，其形態為 list 。
Date = np.array(['03/01', '03/02', '03/03', '03/04', '03/05', '03/07', '03/08',
        '03/09', '03/10', '03/11', '03/12', '03/14', '03/15', '03/16',
        '03/17', '03/18', '03/19','03/21', '03/22', '03/23', '03/24',
        '03/25', '03/26', '03/28', '03/29', '03/30'])

 # x軸的值
Volumes = np.array([10934.5, 12922.5, 15602.5, 18457.5, 20035.0, 26097.5,  20104.0,
            16512.5, 15385.0, 16167.5, 16250.5, 25162.5, 18751.5, 17651.0,
           16036.0, 19933.5, 17460.0, 27273.0, 22006.0, 21021.0, 20627.0,
           25011.0, 22002.0, 26257.0, 15105.0, 14487.0])
 # y軸的值

x = np.sum(Volumes)
len_x = len(Volumes)
avg_volumes = x / len_x
print("平均交易量為: " + str(avg_volumes))



data_df = pd.DataFrame({"Date(Year = 106)": Date, ## 欄位名稱：值
                     "Volume": Volumes ## 欄位名稱：值
                         }
                       )

sns.factorplot(data = data_df, x="Date(Year = 106)", y="Volume", ci = None)
sns.plt.show() ## 顯示出來


```

#以下尚未完成修改

n be run from the command line to get a demo of what it does (and to execute the examples from the book):

```bat
python recommender_systems.py
```  

Additionally, I've collected all the [links](https://github.com/joelgrus/data-science-from-scratch/blob/master/links.md) from the book.

And, by popular demand, I made an index of functions defined in the book, by chapter and page number. 
The data is in a [spreadsheet](https://docs.google.com/spreadsheets/d/1mjGp94ehfxWOEaAFJsPiHqIeOioPH1vN1PdOE6v1az8/edit?usp=sharing), or I also made a toy (experimental) [searchable webapp](http://joelgrus.com/experiments/function-index/).

## Table of Contents

1. Introduction
2. A Crash Course in Python
3. [Visualizing Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/visualizing_data.py)
4. [Linear Algebra](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/linear_algebra.py)
5. [Statistics](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/statistics.py)
6. [Probability](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/probability.py)
7. [Hypothesis and Inference](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/hypothesis_and_inference.py)
8. [Gradient Descent](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/gradient_descent.py)
9. [Getting Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/getting_data.py)
10. [Working With Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/working_with_data.py)
11. [Machine Learning](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/machine_learning.py)
12. [k-Nearest Neighbors](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/nearest_neighbors.py)
13. [Naive Bayes](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/naive_bayes.py)
14. [Simple Linear Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/simple_linear_regression.py)
15. [Multiple Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/multiple_regression.py)
16. [Logistic Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/logistic_regression.py)
17. [Decision Trees](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/decision_trees.py)
18. [Neural Networks](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/neural_networks.py)
19. [Clustering](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/clustering.py)
20. [Natural Language Processing](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/natural_language_processing.py)
21. [Network Analysis](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/network_analysis.py)
22. [Recommender Systems](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/recommender_systems.py)
23. [Databases and SQL](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/databases.py)
24. [MapReduce](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/mapreduce.py)
25. Go Forth And Do Data Science
