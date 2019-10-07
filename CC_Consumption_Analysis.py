#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings(action="ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# In[2]:


data = pd.read_csv("C:/Users/zhang/Desktop/ccdata/CC GENERAL.csv")


# In[3]:


len(data)


# * 数据量为8950

# In[5]:


data.columns


# In[6]:


len(data.columns)


# * 一共18个特征

# 特征的含义：
# * CUST_ID：客户ID
# * BALANCE：余额
# * BALANCE_FREQUENCY：余额变动的频率(0-1,1是频繁更新，0是完全不更新)
# * PURCHASES：购买
# * ONEOFF_PURCHASES：一次性付款
# * INSTALLMENTS_PURCHASES：分期付款
# * CASH_ADVANCE：预付现金
# * PURCHASES_FREQUENCY：购买频率
# * ONEOFF_PURCHASES_FREQUENCY：一次性购买频率
# * PURCHASES_INSTALLMENTS_FREQUENCY：分期付款购买频率
# * CASH_ADVANCE_FREQUENCY：预付现金频率
# * CASH_ADVANCE_TRX：？
# * PURCHASES_TRX：？
# * CREDIT_LIMIT：信用限额
# * PAYMENTS：付款
# * MINIMUM_PAYMENTS：最低付款
# * PRC_FULL_PAYMENT：全额付款
# * TENURE：还款期限

# In[7]:


data.isnull().sum().sort_values(ascending=False)


# * 上面检测了缺失值，发现只有MINIMUM_PAYMENTS和CREDIT_LIMIT中存在缺失值，可以使用均值进行填充

# In[8]:


data['MINIMUM_PAYMENTS'].fillna(data.MINIMUM_PAYMENTS.mean(),inplace=True)
data['CREDIT_LIMIT'].fillna(data.CREDIT_LIMIT.mean(),inplace=True)


# 下面查找异常值

# In[9]:


#显示数据的各种统计量信息
data.describe()


# * 可以看出每个特征的取值范围都是不一样的，而且有的特征之间的取值范围差距是很大的，所以像这样的数值在输入到模型之前需要进行处理，否则会影响到模型的输出结果

# 下面用箱型图

# In[10]:


f,axes = plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(y='BALANCE',data=data,ax=axes[0])
sns.boxplot(y='BALANCE_FREQUENCY',data=data,ax=axes[1])
sns.boxplot(y='PURCHASES',data=data,ax=axes[2])
sns.boxplot(y='ONEOFF_PURCHASES',data=data,ax=axes[3])


# In[11]:


f,axes = plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(y='INSTALLMENTS_PURCHASES',data=data,ax=axes[0])
sns.boxplot(y='CASH_ADVANCE',data=data,ax=axes[1])
sns.boxplot(y='PURCHASES_FREQUENCY',data=data,ax=axes[2])
sns.boxplot(y='ONEOFF_PURCHASES_FREQUENCY',data=data,ax=axes[3])


# In[12]:


f,axes = plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(y='PURCHASES_INSTALLMENTS_FREQUENCY',data=data,ax=axes[0])
sns.boxplot(y='CASH_ADVANCE_FREQUENCY',data=data,ax=axes[1])
sns.boxplot(y='CASH_ADVANCE_TRX',data=data,ax=axes[2])
sns.boxplot(y='PURCHASES_TRX',data=data,ax=axes[3])


# In[13]:


f,axes = plt.subplots(ncols=5,figsize=(20,4))

sns.boxplot(y='CREDIT_LIMIT',data=data,ax=axes[0])
sns.boxplot(y='PAYMENTS',data=data,ax=axes[1])
sns.boxplot(y='MINIMUM_PAYMENTS',data=data,ax=axes[2])
sns.boxplot(y='PRC_FULL_PAYMENT',data=data,ax=axes[3])
sns.boxplot(y='TENURE',data=data,ax=axes[4])


# * 从上面的可视化结果可以看出，各个特征中都存在很多的异常值，但是这些异常值可能都是合理存在的，并不是那种错误的值，所以不应该进行剔除，而是使用转换等方式进行处理

# 根据不同的取值进行不同的转换

# * 对下面这些特征设置不同的区间，然后根据区间范围进行转换处理

# In[14]:


columns=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
         'PAYMENTS', 'MINIMUM_PAYMENTS']

for column in columns:
    new = 'new_' + column
    data[new] = 0
    data.loc[((data[column]>0)&(data[column]<=5000)),new] = 1
    data.loc[((data[column]>5000)&(data[column]<=10000)),new] = 2
    data.loc[((data[column]>10000)),new] = 3


# In[15]:


columns=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
         'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']

for column in columns:
    new = 'new_' + column
    data[new]=0
    data.loc[((data[column]>0)&(data[column]<=0.4)),new] = 1
    data.loc[((data[column]>0.4)&(data[column]<=0.6)),new] = 2
    data.loc[((data[column]>0.6)&(data[column]<=1)),new] = 3


# In[16]:


columns=['PURCHASES_TRX', 'CASH_ADVANCE_TRX']

for column in columns:
    new = 'new_' + column
    data[new]=0
    data.loc[((data[column]>0)&(data[column]<=40)),new] = 1
    data.loc[((data[column]>40)&(data[column]<=60)),new] = 2
    data.loc[((data[column]>60)&(data[column]<=100)),new] = 3


# * 上面都使用了1、2、3作为映射的结果是因为原始数据含有了大小关系，而我这样处理后同样可以保留这种序的含义在里面

# In[17]:


#删除原始数据的特征
data.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
           'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
           'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
           'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
           'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
           'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT' ], axis=1, inplace=True)

X = np.asarray(data)


# In[19]:


#标准化处理，不改变数据形状
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X = scale.fit_transform(X)


# 数据处理结束后就可以带入到模型中进行处理了，而我的目的是对客户进行分群，所以可以使用聚类算法，但是并不能确定要设定多大的簇类数，所以设定了一个范围，然后在指定范围内找出最好的簇类数目

# In[20]:


n_clusters = 30
cost = []
new_cost = []

for i in range(1,n_clusters):
    kmeans = KMeans(i)
    kmeans.fit(X)
    labels = kmeans.labels_
    cost.append(kmeans.inertia_)
    #new_cost.append(metrics.silhouette_score(X,labels,metric='euclidean'))


# In[21]:


plt.plot(cost, 'bx-')


# * 上图是k从1到30聚类模型对应的模型评估指标的变化情况
# * kmeans中的k是可以取任意值的，不过一般要根据实际的要求来确定，或者设定一个取值范围来尝试找到一个比较合理的聚类效果
# * 这里的K最好还是应该根据具体情况进行选择，要根据业务要求选取一个合适的簇数目，这里选一个6

# In[23]:


kmeans = KMeans(6)
kmeans.fit(X)
labels = kmeans.labels_


# In[24]:


df_data = data.copy()
df_data = pd.concat([df_data,pd.DataFrame({'labels':labels})],axis=1)


# 下面分析一下聚类之后的效果到底怎么样

# In[25]:


for column in df_data:
    grid= sns.FacetGrid(df_data,col='labels')
    grid.map(plt.hist,column)


# 上面的结果有一个问题，就是labels为4时对应的数据很少，只有159条数据，而其他类别的数据都是千位级以上的数据，所以我觉得这类数据是不够规模的，也就没有必要单独提取出来进行分析，所以可以尝试将K设定为5在进行聚类

# K为5时的效果依然不好，所以将K设置为4

# In[76]:


kmeans = KMeans(4)
kmeans.fit(X)
labels = kmeans.labels_


# In[77]:


data = pd.concat([data,pd.DataFrame({'labels':labels})],axis=1)


# In[83]:


for column in data:
    grid= sns.FacetGrid(data,col='labels')
    grid.map(plt.hist,column)


# * 上面是针对每个簇进行的可视化，可以从中总结出每个群体的一些特点，不过总结的效果也要根据K的选择来决定，如果K设定的不合理，效果会变差

# label 0：
# * 信用额度比较低，证明这类人群的收入可能并不高；
# * 购物频率比较低，证明不喜爱购物；
# * 不使用分期的方式进行购物
# * 不使用预付消费的方式，可能表名这类人群没有什么固定的购买目标，买的东西会比较杂
# * 总结：收入不高，不是很喜欢购物，且没有明确的购买目标，忠诚度不够

# label0总结：针对客户的购买喜好推荐价格较低或正在促销的产品。

# label 1：
# * 从余额，购买频率和余额变动频率可以看出这类人群收入比较高，且喜欢购物；
# * 从分期使用频率可以看出，这类人群也比较喜欢分期付款的方式；
# * 从预付消费金额和使用频率可以看出，这类人群有使用预付方式进行消费的习惯，说明这类人群更忠诚，喜欢去固定的地点或者习惯的地方进行消费
# * 总结：收入较高，且喜欢购物，可以接受分期和预付的方式进行消费，有固定的消费地点或者区域，客户忠诚度高

# label1总结：针对购买喜好推荐价格更高的产品，并提供分期的消费方式，还可以针对这类客户制定充值优惠等活动，这样既可以促进消费又可以提高客户忠的诚度

# label 2：
# * 总结：这类人群不喜欢购物，不喜欢分期，经常使用的消费方式基本都是预付消费

# label2总结：针对购买喜好推荐相应产品，并提供充值优惠等活动，这样既可以促进消费又可以提高客户忠的诚度

# label 3：
# * 总结：收入高，非常喜欢购物，也经常使用分期的方式，可能是购物狂，但没有预付消费的方式，证明没有明确的固定消费地点或区域，客户忠诚度不够

# label3总结：针对购买喜好推荐价格更高的各类产品，并提供分期付款的方式，并且对于这类客户可以制定充值等活动来吸纳成为会员，这样既可以促进消费又可以提高客户忠的诚度
