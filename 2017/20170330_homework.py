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

## sns.factorplot(data = 資料來源, x = “x 軸的 title”, y = “y 軸的 title”)
sns.factorplot(data = data_df, x="Date(Year = 106)", y="Volume", ci = None)
sns.plt.show() ## 顯示出來

