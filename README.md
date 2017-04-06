倩綾的筆記
=========================

以下是我使用Numpy、Pandas以及Seaborn所做出來的程式碼。(結合第2組和第3組所介紹的功能)。

其資料來源是http://amis.afa.gov.tw/fruit/FruitChartProdTransPriceVolumeTrend.aspx。

查詢台灣全部市場106年1月1日~3月30日的草莓的交易量(KG)。

日期 : 2017/03/30
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

CH06  Probability
```python
# coding=UTF-8
from __future__ import division
from collections import Counter
import math, random
from matplotlib import pyplot as plt

def plot_normal_pdfs(plt):
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
    plt.plot(xs, [normal_cdf(x, sigma=1, mu=3) for x in xs], ':', label='mu=3,sigma=1')
    plt.plot(xs, [normal_cdf(x, mu=-4, sigma=0.5) for x in xs], '-.', label='mu=-1,sigma=1')
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
