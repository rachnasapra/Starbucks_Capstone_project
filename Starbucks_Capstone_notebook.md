
# Starbucks Capstone Challenge

### Introduction

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 

Not all users receive the same offer, and that is the challenge to solve with this data set.

Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 

Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.

### Example

To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.

However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.

### Cleaning

This makes data cleaning especially important and tricky.

You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.

### Final Advice

Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# Data Sets

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

**Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  

You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:

<img src="pic1.png"/>

Then you will want to run the above command:

<img src="pic2.png"/>

Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.


```python
import pandas as pd
import numpy as np
import math
import json
% matplotlib inline

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


```


```python

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
# import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import keras
```

    Using TensorFlow backend.



```python

##Data Analysis :- looking each dataset deeply to understand varibles and outliers 
#Portfolio
#Profile
#Transcript
```


```python
#Knowing about structure
portfolio.head()#(Portfolio have 6 variables: 1. Channel 2.Difficulty 3.Duration 4:id 5:offer_type 6.Reward)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>channels</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>id</th>
      <th>offer_type</th>
      <th>reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[email, mobile, social]</td>
      <td>10</td>
      <td>7</td>
      <td>ae264e3637204a6fb9bb56bc8210ddfd</td>
      <td>bogo</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[web, email, mobile, social]</td>
      <td>10</td>
      <td>5</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>bogo</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[web, email, mobile]</td>
      <td>0</td>
      <td>4</td>
      <td>3f207df678b143eea3cee63160fa8bed</td>
      <td>informational</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[web, email, mobile]</td>
      <td>5</td>
      <td>7</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>bogo</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[web, email]</td>
      <td>20</td>
      <td>10</td>
      <td>0b1e1539f2cc45b7b9fa7c272da2e1d7</td>
      <td>discount</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# data information
portfolio.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 6 columns):
    channels      10 non-null object
    difficulty    10 non-null int64
    duration      10 non-null int64
    id            10 non-null object
    offer_type    10 non-null object
    reward        10 non-null int64
    dtypes: int64(3), object(3)
    memory usage: 560.0+ bytes



```python
#Knowing about structure
profile.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>became_member_on</th>
      <th>gender</th>
      <th>id</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>118</td>
      <td>20170212</td>
      <td>None</td>
      <td>68be06ca386d4c31939f3a4f0e3dd783</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>20170715</td>
      <td>F</td>
      <td>0610b486422d4921ae7d2bf64640c50b</td>
      <td>112000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>118</td>
      <td>20180712</td>
      <td>None</td>
      <td>38fe809add3b4fcf9315a9694bb96ff5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75</td>
      <td>20170509</td>
      <td>F</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>118</td>
      <td>20170804</td>
      <td>None</td>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#it shows we have 6 profiling variable avaiblale (1.Age 2.Become_member_on 3.Gender 4.id 5.Income)
```


```python
#Knowing about structure
transcript.head()#(it has Four varible Event person Time Value)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event</th>
      <th>person</th>
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>offer received</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>0</td>
      <td>{'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>offer received</td>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>0</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>offer received</td>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>0</td>
      <td>{'offer id': '2906b810c7d4411798c6938adc9daaa5'}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>offer received</td>
      <td>8ec6ce2a7e7949b1bf142def7d0e0586</td>
      <td>0</td>
      <td>{'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>offer received</td>
      <td>68617ca6246f4fbc85e91a2a49552598</td>
      <td>0</td>
      <td>{'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}</td>
    </tr>
  </tbody>
</table>
</div>




```python
# describe the dataset
portfolio.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>difficulty</th>
      <th>duration</th>
      <th>reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.700000</td>
      <td>6.500000</td>
      <td>4.200000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.831905</td>
      <td>2.321398</td>
      <td>3.583915</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.500000</td>
      <td>7.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10.000000</td>
      <td>7.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>20.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# describe the dataset
profile.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17000.000000</td>
      <td>1.700000e+04</td>
      <td>14825.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>62.531412</td>
      <td>2.016703e+07</td>
      <td>65404.991568</td>
    </tr>
    <tr>
      <th>std</th>
      <td>26.738580</td>
      <td>1.167750e+04</td>
      <td>21598.299410</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>2.013073e+07</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>45.000000</td>
      <td>2.016053e+07</td>
      <td>49000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>58.000000</td>
      <td>2.017080e+07</td>
      <td>64000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>73.000000</td>
      <td>2.017123e+07</td>
      <td>80000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>118.000000</td>
      <td>2.018073e+07</td>
      <td>120000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# describe the dataset
transcript.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>306534.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>366.382940</td>
    </tr>
    <tr>
      <th>std</th>
      <td>200.326314</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>186.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>408.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>528.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>714.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Rows and colums
portfolio.shape
```




    (10, 6)




```python
#Rows and colums
profile.shape
```




    (17000, 5)




```python
#Rows and colums
transcript.shape
```




    (306534, 4)




```python

# check weather there is a null value or not
#During Analysis we will learn more about each variable but for now we keep in simple and short
portfolio.isnull().sum()
```




    channels      0
    difficulty    0
    duration      0
    id            0
    offer_type    0
    reward        0
    dtype: int64




```python
# check weather there is a null value or not
#During Analysis we will learn more about each variable but for now we keep in simple and short
profile.isnull().sum()
```




    age                    0
    became_member_on       0
    gender              2175
    id                     0
    income              2175
    dtype: int64




```python
# check weather there is a null value or not
#During Analysis we will learn more about each variable but for now we keep in simple and short
transcript.isnull().sum()
```




    event     0
    person    0
    time      0
    value     0
    dtype: int64




```python
# datatype of dataframe
portfolio.dtypes
```




    channels      object
    difficulty     int64
    duration       int64
    id            object
    offer_type    object
    reward         int64
    dtype: object




```python
# datatype of dataframe
profile.dtypes
```




    age                   int64
    became_member_on      int64
    gender               object
    id                   object
    income              float64
    dtype: object




```python
# datatype of dataframe
transcript.dtypes
```




    event     object
    person    object
    time       int64
    value     object
    dtype: object




```python
#Note : In Transacript and Portfolio Dataset no null values are there. But Profile dataset have null values. 
#So we need to clean it and also check whether it contains Outliers.
#lets dig this dataset.

```


```python
profile['gender'].unique()# As we can see None Gender can tells us more about outliers.
```




    array([None, 'F', 'M', 'O'], dtype=object)




```python
profile['income'].unique()# As we can see Nan values can tells us more about outliers.
```




    array([     nan,  112000.,  100000.,   70000.,   53000.,   51000.,
             57000.,   46000.,   71000.,   52000.,   42000.,   40000.,
             69000.,   88000.,   59000.,   41000.,   96000.,   89000.,
             33000.,   68000.,   63000.,   30000.,   98000.,   37000.,
             80000.,   48000.,   38000.,   56000.,   93000.,   50000.,
             35000.,   47000.,   87000.,   76000.,   64000.,   72000.,
            117000.,   55000.,   77000.,   85000.,   36000.,   91000.,
            107000.,   66000.,   58000.,   74000.,   84000.,   54000.,
             49000.,   73000.,   78000.,   31000.,   60000.,   44000.,
            114000.,   65000.,   79000.,   67000.,   94000.,  108000.,
             61000.,   43000.,   92000.,   62000.,   83000.,   34000.,
            105000.,   82000.,  118000.,  109000.,   99000.,   45000.,
            106000.,   95000.,  103000.,  101000.,  110000.,   86000.,
             39000.,   75000.,   90000.,   81000.,   32000.,  120000.,
            119000.,   97000.,  104000.,  113000.,  115000.,  111000.,
            102000.,  116000.])




```python
profile['age'].unique()# As we can see 101, 118 age can tells us more about outliers.
```




    array([118,  55,  75,  68,  65,  58,  61,  26,  62,  49,  57,  40,  64,
            78,  42,  56,  33,  46,  59,  67,  53,  22,  96,  69,  20,  45,
            54,  39,  41,  79,  66,  29,  44,  63,  36,  76,  77,  30,  51,
            27,  73,  74,  70,  89,  50,  90,  60,  19,  72,  52,  18,  71,
            83,  43,  47,  32,  38,  34,  85,  48,  35,  82,  21,  24,  81,
            25,  37,  23, 100,  28,  84,  80,  87,  86,  94,  31,  88,  95,
            93,  91,  92,  98, 101,  97,  99])




```python
# filter = profile["gender"]=="None"
# profile.where(filter)
# profile
```


```python

profile.sort_values(by='age', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>became_member_on</th>
      <th>gender</th>
      <th>id</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>118</td>
      <td>20170212</td>
      <td>None</td>
      <td>68be06ca386d4c31939f3a4f0e3dd783</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3510</th>
      <td>118</td>
      <td>20180331</td>
      <td>None</td>
      <td>a94d4b0cca4d47128920f66a5286431d</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12484</th>
      <td>118</td>
      <td>20160727</td>
      <td>None</td>
      <td>b04385001db14fdf87829c6163ae9ddd</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3483</th>
      <td>118</td>
      <td>20170917</td>
      <td>None</td>
      <td>d4309b7d75174eaa8115f4357b28cd98</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12458</th>
      <td>118</td>
      <td>20180313</td>
      <td>None</td>
      <td>c28414b774094421b34abfe44c6d303c</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3491</th>
      <td>118</td>
      <td>20160421</td>
      <td>None</td>
      <td>f02fa2294c24490db75354d38a1eabad</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3494</th>
      <td>118</td>
      <td>20180414</td>
      <td>None</td>
      <td>da5724d9ac2c45feaa532cdfd4f1ec4a</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3498</th>
      <td>118</td>
      <td>20140512</td>
      <td>None</td>
      <td>5e940120796c4804ae4a5e1f30838acd</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3502</th>
      <td>118</td>
      <td>20150524</td>
      <td>None</td>
      <td>dd37f142a5354f539b2f246cd5a4a6d3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12432</th>
      <td>118</td>
      <td>20150831</td>
      <td>None</td>
      <td>e1c5485f813a47a984401e3f8e618d40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12425</th>
      <td>118</td>
      <td>20180305</td>
      <td>None</td>
      <td>f74c63ee851140e195e36f802fce118c</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12423</th>
      <td>118</td>
      <td>20160627</td>
      <td>None</td>
      <td>d59b9601634a4116bb65209fe2a47af5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12410</th>
      <td>118</td>
      <td>20170910</td>
      <td>None</td>
      <td>53efe838339f4ff69b567c3bbad9d9d8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3458</th>
      <td>118</td>
      <td>20170408</td>
      <td>None</td>
      <td>597760e0272a460cbc840b471d768427</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3516</th>
      <td>118</td>
      <td>20180101</td>
      <td>None</td>
      <td>03ae3b023c2540f4969b64b633be5ede</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12403</th>
      <td>118</td>
      <td>20180303</td>
      <td>None</td>
      <td>cc82a0a1f09d40c1ac79d5b1d844cc9b</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3521</th>
      <td>118</td>
      <td>20180517</td>
      <td>None</td>
      <td>75da6707b8824502852123cf9e6d353e</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12396</th>
      <td>118</td>
      <td>20171019</td>
      <td>None</td>
      <td>4ffaf40fceb94983a3bb8f92c069bfc8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3525</th>
      <td>118</td>
      <td>20180624</td>
      <td>None</td>
      <td>f04c9064ecfe4852991fc510428c7e5b</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3529</th>
      <td>118</td>
      <td>20171230</td>
      <td>None</td>
      <td>392bd914245349489e4972fb7e4ce86d</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12392</th>
      <td>118</td>
      <td>20180630</td>
      <td>None</td>
      <td>e759379af62b40dea7c1c707b1e49a2a</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12383</th>
      <td>118</td>
      <td>20150910</td>
      <td>None</td>
      <td>d2925895f1094ecb99cf24d0a62ce1c2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3542</th>
      <td>118</td>
      <td>20161204</td>
      <td>None</td>
      <td>8f8e6021ca784840b5c77a7e9647c751</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3558</th>
      <td>118</td>
      <td>20151021</td>
      <td>None</td>
      <td>56b1a70016c74f6db9376a33cb423dd5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3463</th>
      <td>118</td>
      <td>20160729</td>
      <td>None</td>
      <td>253154f26d2c4201aaed2192bdbbd8dc</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12502</th>
      <td>118</td>
      <td>20170520</td>
      <td>None</td>
      <td>e0b8743326fc4ce99677bcc4d3dba3a5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12355</th>
      <td>118</td>
      <td>20170322</td>
      <td>None</td>
      <td>7a732c740d84488f99fbbaa55b6c144c</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12535</th>
      <td>118</td>
      <td>20180624</td>
      <td>None</td>
      <td>55a1796bd57140279edac3bb4166a8cb</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3393</th>
      <td>118</td>
      <td>20170802</td>
      <td>None</td>
      <td>c4196572916f4e7a8a5e6b0c18c7dd40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12572</th>
      <td>118</td>
      <td>20161021</td>
      <td>None</td>
      <td>ed147cbcfca5407c9965785f29d34a70</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14253</th>
      <td>18</td>
      <td>20180725</td>
      <td>M</td>
      <td>717eee391dc44e9aba4315b917ebaa35</td>
      <td>45000.0</td>
    </tr>
    <tr>
      <th>16922</th>
      <td>18</td>
      <td>20151219</td>
      <td>F</td>
      <td>379ef2e7156d4e339bc81b29e38548f6</td>
      <td>31000.0</td>
    </tr>
    <tr>
      <th>6985</th>
      <td>18</td>
      <td>20150812</td>
      <td>M</td>
      <td>d15e46c18322472ab7e3dfbc2951aefa</td>
      <td>49000.0</td>
    </tr>
    <tr>
      <th>5906</th>
      <td>18</td>
      <td>20150930</td>
      <td>M</td>
      <td>2ba1d65405594702af723081531011ef</td>
      <td>38000.0</td>
    </tr>
    <tr>
      <th>4225</th>
      <td>18</td>
      <td>20130808</td>
      <td>M</td>
      <td>2825186736d4433c98653bcea5957cb3</td>
      <td>56000.0</td>
    </tr>
    <tr>
      <th>12781</th>
      <td>18</td>
      <td>20171226</td>
      <td>M</td>
      <td>976b4e8c34884e0bb9af62e5952299af</td>
      <td>43000.0</td>
    </tr>
    <tr>
      <th>12057</th>
      <td>18</td>
      <td>20171208</td>
      <td>M</td>
      <td>2a01fc19e5c942efa247588405f43b7a</td>
      <td>51000.0</td>
    </tr>
    <tr>
      <th>6763</th>
      <td>18</td>
      <td>20180522</td>
      <td>F</td>
      <td>4e3afc3a5bd54635b938093420edf973</td>
      <td>46000.0</td>
    </tr>
    <tr>
      <th>9066</th>
      <td>18</td>
      <td>20180325</td>
      <td>M</td>
      <td>5cb68688b66b42db8d1985340c289eb7</td>
      <td>48000.0</td>
    </tr>
    <tr>
      <th>3246</th>
      <td>18</td>
      <td>20180313</td>
      <td>M</td>
      <td>6ad061a0de7049a194735a19eff2279d</td>
      <td>51000.0</td>
    </tr>
    <tr>
      <th>13791</th>
      <td>18</td>
      <td>20180112</td>
      <td>M</td>
      <td>25af91e1d87248c387b17bcba16372b8</td>
      <td>31000.0</td>
    </tr>
    <tr>
      <th>5456</th>
      <td>18</td>
      <td>20161229</td>
      <td>M</td>
      <td>abdc98fc32234a17b962e035422289fd</td>
      <td>47000.0</td>
    </tr>
    <tr>
      <th>7006</th>
      <td>18</td>
      <td>20151025</td>
      <td>F</td>
      <td>13f95724ed4042aaa8953e8ed5d073c9</td>
      <td>43000.0</td>
    </tr>
    <tr>
      <th>12973</th>
      <td>18</td>
      <td>20170705</td>
      <td>M</td>
      <td>f0a3a3c05e3c4e2e84929a49a6b5488c</td>
      <td>54000.0</td>
    </tr>
    <tr>
      <th>500</th>
      <td>18</td>
      <td>20150503</td>
      <td>M</td>
      <td>1a17eef552164fa79276342fc46c9364</td>
      <td>48000.0</td>
    </tr>
    <tr>
      <th>11470</th>
      <td>18</td>
      <td>20170115</td>
      <td>M</td>
      <td>fcdc9e86b659499d882a4714ea53e974</td>
      <td>45000.0</td>
    </tr>
    <tr>
      <th>4340</th>
      <td>18</td>
      <td>20170907</td>
      <td>M</td>
      <td>49fc32bb19f24c608a04b573d183bf0d</td>
      <td>56000.0</td>
    </tr>
    <tr>
      <th>14805</th>
      <td>18</td>
      <td>20171118</td>
      <td>M</td>
      <td>095bc1a9b5f64d0f88ed616df292a3ec</td>
      <td>55000.0</td>
    </tr>
    <tr>
      <th>2905</th>
      <td>18</td>
      <td>20170811</td>
      <td>M</td>
      <td>e2966f15389e4e0e9a5891bd675a5f11</td>
      <td>72000.0</td>
    </tr>
    <tr>
      <th>750</th>
      <td>18</td>
      <td>20161031</td>
      <td>M</td>
      <td>d62cf96d9f19459e8ddb11720f1fa78c</td>
      <td>36000.0</td>
    </tr>
    <tr>
      <th>12636</th>
      <td>18</td>
      <td>20150730</td>
      <td>M</td>
      <td>4affa3ba745a4d2ab94abe96d963d3ac</td>
      <td>68000.0</td>
    </tr>
    <tr>
      <th>3712</th>
      <td>18</td>
      <td>20141127</td>
      <td>F</td>
      <td>e0749ba1296b41e9adbdc27c979b2510</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>6743</th>
      <td>18</td>
      <td>20180725</td>
      <td>M</td>
      <td>2f31ca6bc8f741f2b1a809a38971d6bb</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>1272</th>
      <td>18</td>
      <td>20180421</td>
      <td>M</td>
      <td>5549b705af144f9aa6ff76a1515fc908</td>
      <td>46000.0</td>
    </tr>
    <tr>
      <th>15591</th>
      <td>18</td>
      <td>20170911</td>
      <td>F</td>
      <td>eb695e77fdb84f359f7de63eabe6bc5f</td>
      <td>31000.0</td>
    </tr>
    <tr>
      <th>16039</th>
      <td>18</td>
      <td>20151209</td>
      <td>M</td>
      <td>c17fdcb8833f4759bb2854aeb594ff6d</td>
      <td>58000.0</td>
    </tr>
    <tr>
      <th>12947</th>
      <td>18</td>
      <td>20150914</td>
      <td>M</td>
      <td>6bd1af2841a6412ebce35605cc4c394b</td>
      <td>60000.0</td>
    </tr>
    <tr>
      <th>14291</th>
      <td>18</td>
      <td>20160217</td>
      <td>M</td>
      <td>6bf6090ce6bf4dfa8c572d3f5fb1acc9</td>
      <td>49000.0</td>
    </tr>
    <tr>
      <th>10307</th>
      <td>18</td>
      <td>20180417</td>
      <td>M</td>
      <td>5a52e27405c84e58a0d7414562df9e10</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>4016</th>
      <td>18</td>
      <td>20171010</td>
      <td>M</td>
      <td>68012cc999eb42fda88b5f9b54603d9c</td>
      <td>67000.0</td>
    </tr>
  </tbody>
</table>
<p>17000 rows × 5 columns</p>
</div>




```python
# check for age

profile.sort_values(by='age', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>became_member_on</th>
      <th>gender</th>
      <th>id</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>118</td>
      <td>20170212</td>
      <td>None</td>
      <td>68be06ca386d4c31939f3a4f0e3dd783</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3510</th>
      <td>118</td>
      <td>20180331</td>
      <td>None</td>
      <td>a94d4b0cca4d47128920f66a5286431d</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12484</th>
      <td>118</td>
      <td>20160727</td>
      <td>None</td>
      <td>b04385001db14fdf87829c6163ae9ddd</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3483</th>
      <td>118</td>
      <td>20170917</td>
      <td>None</td>
      <td>d4309b7d75174eaa8115f4357b28cd98</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12458</th>
      <td>118</td>
      <td>20180313</td>
      <td>None</td>
      <td>c28414b774094421b34abfe44c6d303c</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3491</th>
      <td>118</td>
      <td>20160421</td>
      <td>None</td>
      <td>f02fa2294c24490db75354d38a1eabad</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3494</th>
      <td>118</td>
      <td>20180414</td>
      <td>None</td>
      <td>da5724d9ac2c45feaa532cdfd4f1ec4a</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3498</th>
      <td>118</td>
      <td>20140512</td>
      <td>None</td>
      <td>5e940120796c4804ae4a5e1f30838acd</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3502</th>
      <td>118</td>
      <td>20150524</td>
      <td>None</td>
      <td>dd37f142a5354f539b2f246cd5a4a6d3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12432</th>
      <td>118</td>
      <td>20150831</td>
      <td>None</td>
      <td>e1c5485f813a47a984401e3f8e618d40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12425</th>
      <td>118</td>
      <td>20180305</td>
      <td>None</td>
      <td>f74c63ee851140e195e36f802fce118c</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12423</th>
      <td>118</td>
      <td>20160627</td>
      <td>None</td>
      <td>d59b9601634a4116bb65209fe2a47af5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12410</th>
      <td>118</td>
      <td>20170910</td>
      <td>None</td>
      <td>53efe838339f4ff69b567c3bbad9d9d8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3458</th>
      <td>118</td>
      <td>20170408</td>
      <td>None</td>
      <td>597760e0272a460cbc840b471d768427</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3516</th>
      <td>118</td>
      <td>20180101</td>
      <td>None</td>
      <td>03ae3b023c2540f4969b64b633be5ede</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12403</th>
      <td>118</td>
      <td>20180303</td>
      <td>None</td>
      <td>cc82a0a1f09d40c1ac79d5b1d844cc9b</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3521</th>
      <td>118</td>
      <td>20180517</td>
      <td>None</td>
      <td>75da6707b8824502852123cf9e6d353e</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12396</th>
      <td>118</td>
      <td>20171019</td>
      <td>None</td>
      <td>4ffaf40fceb94983a3bb8f92c069bfc8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3525</th>
      <td>118</td>
      <td>20180624</td>
      <td>None</td>
      <td>f04c9064ecfe4852991fc510428c7e5b</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3529</th>
      <td>118</td>
      <td>20171230</td>
      <td>None</td>
      <td>392bd914245349489e4972fb7e4ce86d</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12392</th>
      <td>118</td>
      <td>20180630</td>
      <td>None</td>
      <td>e759379af62b40dea7c1c707b1e49a2a</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12383</th>
      <td>118</td>
      <td>20150910</td>
      <td>None</td>
      <td>d2925895f1094ecb99cf24d0a62ce1c2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3542</th>
      <td>118</td>
      <td>20161204</td>
      <td>None</td>
      <td>8f8e6021ca784840b5c77a7e9647c751</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3558</th>
      <td>118</td>
      <td>20151021</td>
      <td>None</td>
      <td>56b1a70016c74f6db9376a33cb423dd5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3463</th>
      <td>118</td>
      <td>20160729</td>
      <td>None</td>
      <td>253154f26d2c4201aaed2192bdbbd8dc</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12502</th>
      <td>118</td>
      <td>20170520</td>
      <td>None</td>
      <td>e0b8743326fc4ce99677bcc4d3dba3a5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12355</th>
      <td>118</td>
      <td>20170322</td>
      <td>None</td>
      <td>7a732c740d84488f99fbbaa55b6c144c</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12535</th>
      <td>118</td>
      <td>20180624</td>
      <td>None</td>
      <td>55a1796bd57140279edac3bb4166a8cb</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3393</th>
      <td>118</td>
      <td>20170802</td>
      <td>None</td>
      <td>c4196572916f4e7a8a5e6b0c18c7dd40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12572</th>
      <td>118</td>
      <td>20161021</td>
      <td>None</td>
      <td>ed147cbcfca5407c9965785f29d34a70</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14253</th>
      <td>18</td>
      <td>20180725</td>
      <td>M</td>
      <td>717eee391dc44e9aba4315b917ebaa35</td>
      <td>45000.0</td>
    </tr>
    <tr>
      <th>16922</th>
      <td>18</td>
      <td>20151219</td>
      <td>F</td>
      <td>379ef2e7156d4e339bc81b29e38548f6</td>
      <td>31000.0</td>
    </tr>
    <tr>
      <th>6985</th>
      <td>18</td>
      <td>20150812</td>
      <td>M</td>
      <td>d15e46c18322472ab7e3dfbc2951aefa</td>
      <td>49000.0</td>
    </tr>
    <tr>
      <th>5906</th>
      <td>18</td>
      <td>20150930</td>
      <td>M</td>
      <td>2ba1d65405594702af723081531011ef</td>
      <td>38000.0</td>
    </tr>
    <tr>
      <th>4225</th>
      <td>18</td>
      <td>20130808</td>
      <td>M</td>
      <td>2825186736d4433c98653bcea5957cb3</td>
      <td>56000.0</td>
    </tr>
    <tr>
      <th>12781</th>
      <td>18</td>
      <td>20171226</td>
      <td>M</td>
      <td>976b4e8c34884e0bb9af62e5952299af</td>
      <td>43000.0</td>
    </tr>
    <tr>
      <th>12057</th>
      <td>18</td>
      <td>20171208</td>
      <td>M</td>
      <td>2a01fc19e5c942efa247588405f43b7a</td>
      <td>51000.0</td>
    </tr>
    <tr>
      <th>6763</th>
      <td>18</td>
      <td>20180522</td>
      <td>F</td>
      <td>4e3afc3a5bd54635b938093420edf973</td>
      <td>46000.0</td>
    </tr>
    <tr>
      <th>9066</th>
      <td>18</td>
      <td>20180325</td>
      <td>M</td>
      <td>5cb68688b66b42db8d1985340c289eb7</td>
      <td>48000.0</td>
    </tr>
    <tr>
      <th>3246</th>
      <td>18</td>
      <td>20180313</td>
      <td>M</td>
      <td>6ad061a0de7049a194735a19eff2279d</td>
      <td>51000.0</td>
    </tr>
    <tr>
      <th>13791</th>
      <td>18</td>
      <td>20180112</td>
      <td>M</td>
      <td>25af91e1d87248c387b17bcba16372b8</td>
      <td>31000.0</td>
    </tr>
    <tr>
      <th>5456</th>
      <td>18</td>
      <td>20161229</td>
      <td>M</td>
      <td>abdc98fc32234a17b962e035422289fd</td>
      <td>47000.0</td>
    </tr>
    <tr>
      <th>7006</th>
      <td>18</td>
      <td>20151025</td>
      <td>F</td>
      <td>13f95724ed4042aaa8953e8ed5d073c9</td>
      <td>43000.0</td>
    </tr>
    <tr>
      <th>12973</th>
      <td>18</td>
      <td>20170705</td>
      <td>M</td>
      <td>f0a3a3c05e3c4e2e84929a49a6b5488c</td>
      <td>54000.0</td>
    </tr>
    <tr>
      <th>500</th>
      <td>18</td>
      <td>20150503</td>
      <td>M</td>
      <td>1a17eef552164fa79276342fc46c9364</td>
      <td>48000.0</td>
    </tr>
    <tr>
      <th>11470</th>
      <td>18</td>
      <td>20170115</td>
      <td>M</td>
      <td>fcdc9e86b659499d882a4714ea53e974</td>
      <td>45000.0</td>
    </tr>
    <tr>
      <th>4340</th>
      <td>18</td>
      <td>20170907</td>
      <td>M</td>
      <td>49fc32bb19f24c608a04b573d183bf0d</td>
      <td>56000.0</td>
    </tr>
    <tr>
      <th>14805</th>
      <td>18</td>
      <td>20171118</td>
      <td>M</td>
      <td>095bc1a9b5f64d0f88ed616df292a3ec</td>
      <td>55000.0</td>
    </tr>
    <tr>
      <th>2905</th>
      <td>18</td>
      <td>20170811</td>
      <td>M</td>
      <td>e2966f15389e4e0e9a5891bd675a5f11</td>
      <td>72000.0</td>
    </tr>
    <tr>
      <th>750</th>
      <td>18</td>
      <td>20161031</td>
      <td>M</td>
      <td>d62cf96d9f19459e8ddb11720f1fa78c</td>
      <td>36000.0</td>
    </tr>
    <tr>
      <th>12636</th>
      <td>18</td>
      <td>20150730</td>
      <td>M</td>
      <td>4affa3ba745a4d2ab94abe96d963d3ac</td>
      <td>68000.0</td>
    </tr>
    <tr>
      <th>3712</th>
      <td>18</td>
      <td>20141127</td>
      <td>F</td>
      <td>e0749ba1296b41e9adbdc27c979b2510</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>6743</th>
      <td>18</td>
      <td>20180725</td>
      <td>M</td>
      <td>2f31ca6bc8f741f2b1a809a38971d6bb</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>1272</th>
      <td>18</td>
      <td>20180421</td>
      <td>M</td>
      <td>5549b705af144f9aa6ff76a1515fc908</td>
      <td>46000.0</td>
    </tr>
    <tr>
      <th>15591</th>
      <td>18</td>
      <td>20170911</td>
      <td>F</td>
      <td>eb695e77fdb84f359f7de63eabe6bc5f</td>
      <td>31000.0</td>
    </tr>
    <tr>
      <th>16039</th>
      <td>18</td>
      <td>20151209</td>
      <td>M</td>
      <td>c17fdcb8833f4759bb2854aeb594ff6d</td>
      <td>58000.0</td>
    </tr>
    <tr>
      <th>12947</th>
      <td>18</td>
      <td>20150914</td>
      <td>M</td>
      <td>6bd1af2841a6412ebce35605cc4c394b</td>
      <td>60000.0</td>
    </tr>
    <tr>
      <th>14291</th>
      <td>18</td>
      <td>20160217</td>
      <td>M</td>
      <td>6bf6090ce6bf4dfa8c572d3f5fb1acc9</td>
      <td>49000.0</td>
    </tr>
    <tr>
      <th>10307</th>
      <td>18</td>
      <td>20180417</td>
      <td>M</td>
      <td>5a52e27405c84e58a0d7414562df9e10</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>4016</th>
      <td>18</td>
      <td>20171010</td>
      <td>M</td>
      <td>68012cc999eb42fda88b5f9b54603d9c</td>
      <td>67000.0</td>
    </tr>
  </tbody>
</table>
<p>17000 rows × 5 columns</p>
</div>




```python
profile[['gender', "age", "income"]][profile['age']>=100] .head(2500)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>39</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>44</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>45</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>54</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>56</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>57</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>80</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>84</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>96</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>103</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>104</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>108</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>121</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>122</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>128</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>143</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>169</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16829</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16831</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16835</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16839</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16842</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16844</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16845</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16852</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16853</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16856</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16861</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16864</th>
      <td>F</td>
      <td>101</td>
      <td>82000.0</td>
    </tr>
    <tr>
      <th>16869</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16875</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16877</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16885</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16906</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16915</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16923</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16931</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16942</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16951</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16953</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16969</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16977</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16980</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16982</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16989</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16991</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16994</th>
      <td>None</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2192 rows × 3 columns</p>
</div>




```python
profile[['gender', "age", "income"]][profile['age']==101] .head(2500)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1556</th>
      <td>F</td>
      <td>101</td>
      <td>43000.0</td>
    </tr>
    <tr>
      <th>4100</th>
      <td>F</td>
      <td>101</td>
      <td>99000.0</td>
    </tr>
    <tr>
      <th>14846</th>
      <td>F</td>
      <td>101</td>
      <td>56000.0</td>
    </tr>
    <tr>
      <th>15800</th>
      <td>F</td>
      <td>101</td>
      <td>59000.0</td>
    </tr>
    <tr>
      <th>16864</th>
      <td>F</td>
      <td>101</td>
      <td>82000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#it clearly shows age group 118 is outlier as 101 have female gender corresponing to it.
#letsplot graph to see whether its left skwewwed or right skwewed
```


```python
plt.figure()
sns.distplot(profile['age'], bins=50, kde=False);
```


![png](output_32_0.png)



```python
plt.boxplot(profile['age'],vert=0,patch_artist=True)
plt.show()

```


![png](output_33_0.png)



```python
## Data Processing plus Data cleaning
#here we will do some scaling which will helps us during model building. So that fields need transformation.
#Also we will rename some fields and create some dataframe from poriginal data which will further helps in analyis


```


```python
#Created datasets copies as we can rename and format the fields required for analysis,
#it always good practice to make copy of data  because it that you always can go back and 
#see you orginal data incase your formating or any tings mislead.
df_portfolio= portfolio.copy()
df_profile = profile.copy()
df_transcript = transcript.copy()
```


```python
df_portfolio= portfolio.copy()
#DF_portfolio is handled for making numerical variables using key value pair for better handling dat during MODEL built.
# change duration column in hours and rename it to duration_hours
df_portfolio['duration'] = df_portfolio['duration']*24
df_portfolio.rename(columns={'duration':'duration_hours'}, inplace=True)

# Vraibles are converted to numeric fields
scaler = MinMaxScaler() 
numerical = ['difficulty','reward']
df_portfolio[numerical] = scaler.fit_transform(df_portfolio[numerical])
offer_ids = df_portfolio['id'].astype('category').cat.categories.tolist()
new_offer_ids = {'id' : {k: v for k,v in zip(offer_ids,list(range(1,len(offer_ids)+1)))}}
df_portfolio.replace(new_offer_ids, inplace=True)
df_portfolio.rename(columns={'id':'offer_id'}, inplace=True)
offer_types = df_portfolio['offer_type'].astype('category').cat.categories.tolist()
new_offer_types = {'offer_type' : {k: v for k,v in zip(offer_types,list(range(1,len(offer_types)+1)))}}
df_portfolio.replace(new_offer_types, inplace=True)


# split channel into diffrent column, to look multichannel segmentaion 

df_portfolio['Web_channel'] =  df_portfolio['channels'].apply(lambda x: 1 if 'Web' in x else 0)
df_portfolio['Email_channel'] = df_portfolio['channels'].apply(lambda x: 1 if 'email' in x else 0)
df_portfolio['Social_channel'] = df_portfolio['channels'].apply(lambda x: 1 if 'social' in x else 0)
df_portfolio['Mobile_channel'] = df_portfolio['channels'].apply(lambda x: 1 if 'mobile' in x else 0)
# drop the column channel
df_portfolio.drop('channels', axis=1, inplace=True)
df_portfolio.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>difficulty</th>
      <th>duration_hours</th>
      <th>offer_id</th>
      <th>offer_type</th>
      <th>reward</th>
      <th>Web_channel</th>
      <th>Email_channel</th>
      <th>Social_channel</th>
      <th>Mobile_channel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.50</td>
      <td>168</td>
      <td>8</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.50</td>
      <td>120</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00</td>
      <td>96</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.25</td>
      <td>168</td>
      <td>7</td>
      <td>1</td>
      <td>0.5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.00</td>
      <td>240</td>
      <td>1</td>
      <td>2</td>
      <td>0.5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_portfolio.shape
```




    (10, 9)




```python
df_profile = profile.copy()
```


```python

#DF_profile is handled for making numerical variables using key value pair for better handling dat during MODEL built.
customer_ids = df_profile['id'].astype('category').cat.categories.tolist()
new_customer_ids = {'id' : {k: v for k,v in zip(customer_ids,list(range(1,len(customer_ids)+1)))}}
df_profile.replace(new_customer_ids, inplace=True)

df_profile.rename(columns={'id':'customer_id'},inplace=True)

#Age Segments
df_profile['age_group'] = pd.cut(df_profile['age'], bins=[17, 34, 51,70, 103],labels=['Gen_Z', 'Gen_Y ', 'Gen_X', 'Baby_Boom'])

#Income Segmentation
df_profile['income_range'] = pd.cut(df_profile['income'], bins=[29999, 60000, 90000, 120001],labels=['Low', 'Medium', 'high'])


genders = df_profile['gender'].astype('category').cat.categories.tolist()
new_map_gender = {'gender' : {k: v for k,v in zip(genders,list(range(1,len(genders)+1)))}}
df_profile.replace(new_map_gender, inplace=True)
df_profile.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>became_member_on</th>
      <th>gender</th>
      <th>customer_id</th>
      <th>income</th>
      <th>age_group</th>
      <th>income_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>118</td>
      <td>20170212</td>
      <td>NaN</td>
      <td>6962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>20170715</td>
      <td>1.0</td>
      <td>399</td>
      <td>112000.0</td>
      <td>Gen_X</td>
      <td>high</td>
    </tr>
    <tr>
      <th>2</th>
      <td>118</td>
      <td>20180712</td>
      <td>NaN</td>
      <td>3747</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75</td>
      <td>20170509</td>
      <td>1.0</td>
      <td>7997</td>
      <td>100000.0</td>
      <td>Baby_Boom</td>
      <td>high</td>
    </tr>
    <tr>
      <th>4</th>
      <td>118</td>
      <td>20170804</td>
      <td>NaN</td>
      <td>10736</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_profile = profile.copy()
#DF_profile is handled for making numerical variables using key value pair for better handling dat during MODEL built.
customer_ids = df_profile['id'].astype('category').cat.categories.tolist()
new_customer_ids = {'id' : {k: v for k,v in zip(customer_ids,list(range(1,len(customer_ids)+1)))}}
df_profile.replace(new_customer_ids, inplace=True)

df_profile.rename(columns={'id':'customer_id'},inplace=True)

#Age Segments
df_profile['age_group'] = pd.cut(df_profile['age'], bins=[17, 34, 51,70, 103],labels=['Gen_Z', 'Gen_Y ', 'Gen_X', 'Baby_Boom'])
#Converting Age group into numeric numbers
age_groups = df_profile['age_group'].astype('category').cat.categories.tolist()
new_age_groups = {'age_group' : {k: v for k,v in zip(age_groups,list(range(1,len(age_groups)+1)))}}
df_profile.replace(new_age_groups, inplace=True)
#Income Segmentation
df_profile['income_range'] = pd.cut(df_profile['income'], bins=[29999, 60000, 90000, 120001],labels=['Low', 'Medium', 'high'])
income_ranges = df_profile['income_range'].astype('category').cat.categories.tolist()
income_ranges = {'income_range' : {k: v for k,v in zip(income_ranges,list(range(1,len(income_ranges)+1)))}}
df_profile.replace(income_ranges, inplace=True)

genders = df_profile['gender'].astype('category').cat.categories.tolist()
df_profile['became_member_on'] = pd.to_datetime(df_profile['became_member_on'], format='%Y%m%d')
df_profile['year'] = df_profile['became_member_on'].apply(lambda x: str(x)[:4])
new_map_gender = {'gender' : {k: v for k,v in zip(genders,list(range(1,len(genders)+1)))}}
df_profile.replace(new_map_gender, inplace=True)
df_profile.drop(columns = ['age','income'], axis=1, inplace=True)
df_profile.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>became_member_on</th>
      <th>gender</th>
      <th>customer_id</th>
      <th>age_group</th>
      <th>income_range</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-02-12</td>
      <td>NaN</td>
      <td>6962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-07-15</td>
      <td>1.0</td>
      <td>399</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-07-12</td>
      <td>NaN</td>
      <td>3747</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-05-09</td>
      <td>1.0</td>
      <td>7997</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-08-04</td>
      <td>NaN</td>
      <td>10736</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_age_groups
```




    {'age_group': {'Gen_Z': 1, 'Gen_Y ': 2, 'Gen_X': 3, 'Baby_Boom': 4}}




```python
income_ranges
```




    {'income_range': {'Low': 1, 'Medium': 2, 'high': 3}}




```python
profile.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>became_member_on</th>
      <th>gender</th>
      <th>id</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>118</td>
      <td>20170212</td>
      <td>None</td>
      <td>68be06ca386d4c31939f3a4f0e3dd783</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>20170715</td>
      <td>F</td>
      <td>0610b486422d4921ae7d2bf64640c50b</td>
      <td>112000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>118</td>
      <td>20180712</td>
      <td>None</td>
      <td>38fe809add3b4fcf9315a9694bb96ff5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75</td>
      <td>20170509</td>
      <td>F</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>118</td>
      <td>20170804</td>
      <td>None</td>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_transcript= transcript.copy()
# df_transcript.rename(columns={'person':'customer_id'},inplace=True)
new_customer_ids = {'person' : {k: v for k,v in zip(customer_ids,list(range(1,len(customer_ids)+1)))}}
df_transcript.replace(new_customer_ids, inplace=True)
df_transcript['days'] = df_transcript['time'].apply(lambda x: int(x / 24) + (x % 24 > 0))
df_transcript['offer_ids'] = df_transcript['value'].apply(lambda x: x['offer id'] if 'offer id' in x else x['offer_id'] if 'offer_id' in x else np.nan)
offer_ids = df_transcript['offer_ids'].astype('category').cat.categories.tolist()
new_offer_id = {'offer_ids' : {k: v for k,v in zip(offer_ids,list(range(1,len(offer_ids)+1)))}}
df_transcript.replace(new_offer_id, inplace=True)
df_transcript['amount'] = df_transcript['value'].apply(lambda x: x['amount']  if 'amount' in x else x['amount'] if 'amount' in x else 0)
df_transcript['reward'] = df_transcript['value'].apply(lambda x: x['reward']  if 'reward' in x else x['reward'] if 'reward' in x else 0)
events = df_transcript['event'].astype('category').cat.categories.tolist()
new_map_events = {'event' : {k: v for k,v in zip(events,list(range(1,len(events)+1)))}}


df_transcript.replace(new_map_events, inplace=True)
df_transcript.drop(['value'], axis=1, inplace=True)
df_transcript.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event</th>
      <th>person</th>
      <th>time</th>
      <th>days</th>
      <th>offer_ids</th>
      <th>amount</th>
      <th>reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>7997</td>
      <td>0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>10736</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>15044</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>9525</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>6940</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_map_events
```




    {'event': {'offer completed': 1,
      'offer received': 2,
      'offer viewed': 3,
      'transaction': 4}}




```python
new_map_events
```




    {'event': {'offer completed': 1,
      'offer received': 2,
      'offer viewed': 3,
      'transaction': 4}}




```python
new_offer_ids
```




    {'id': {'0b1e1539f2cc45b7b9fa7c272da2e1d7': 1,
      '2298d6c36e964ae4a3e7e9706d1fb8c2': 2,
      '2906b810c7d4411798c6938adc9daaa5': 3,
      '3f207df678b143eea3cee63160fa8bed': 4,
      '4d5c57ea9a6940dd891ad53e9dbe8da0': 5,
      '5a8bc65990b245e5a138643cd4eb9837': 6,
      '9b98b8c7a33c4b65b9aebfe6a799e6d9': 7,
      'ae264e3637204a6fb9bb56bc8210ddfd': 8,
      'f19421c1d4aa40978ebb69ca19b0e20d': 9,
      'fafdcd668e3743c1bb461111dcafc2a4': 10}}




```python
new_offer_id
```




    {'offer_ids': {'0b1e1539f2cc45b7b9fa7c272da2e1d7': 1,
      '2298d6c36e964ae4a3e7e9706d1fb8c2': 2,
      '2906b810c7d4411798c6938adc9daaa5': 3,
      '3f207df678b143eea3cee63160fa8bed': 4,
      '4d5c57ea9a6940dd891ad53e9dbe8da0': 5,
      '5a8bc65990b245e5a138643cd4eb9837': 6,
      '9b98b8c7a33c4b65b9aebfe6a799e6d9': 7,
      'ae264e3637204a6fb9bb56bc8210ddfd': 8,
      'f19421c1d4aa40978ebb69ca19b0e20d': 9,
      'fafdcd668e3743c1bb461111dcafc2a4': 10}}




```python
transcript.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event</th>
      <th>person</th>
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>offer received</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>0</td>
      <td>{'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>offer received</td>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>0</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>offer received</td>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>0</td>
      <td>{'offer id': '2906b810c7d4411798c6938adc9daaa5'}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>offer received</td>
      <td>8ec6ce2a7e7949b1bf142def7d0e0586</td>
      <td>0</td>
      <td>{'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>offer received</td>
      <td>68617ca6246f4fbc85e91a2a49552598</td>
      <td>0</td>
      <td>{'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}</td>
    </tr>
  </tbody>
</table>
</div>




```python
# In above Processing and cleaning steps we have changed data type for many varibles are per needed in Model builts but Udirng data Analyis we will surely name that variable in string 
#to give better understanding.
```


```python
df_profile.dtypes
```




    became_member_on    datetime64[ns]
    gender                     float64
    customer_id                  int64
    age_group                  float64
    income_range               float64
    year                        object
    dtype: object




```python
df_transcript.dtypes
```




    event          int64
    person         int64
    time           int64
    days           int64
    offer_ids    float64
    amount       float64
    reward         int64
    dtype: object




```python
df_portfolio.dtypes
```




    difficulty        float64
    duration_hours      int64
    offer_id            int64
    offer_type          int64
    reward            float64
    Web_channel         int64
    Email_channel       int64
    Social_channel      int64
    Mobile_channel      int64
    dtype: object




```python
#Merging Data together is most valuable part
merge_df1 = df_transcript.merge(df_profile, left_on='person', right_on='customer_id', how='left')

# merge_df1 =df_portfolio.merge(df_transcript,left_on='customer_id', right_on='person',how='outer')
final_merge= merge_df1.merge(df_portfolio,  left_on='offer_ids', right_on='offer_id',how ='left')
```


```python
merge_df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event</th>
      <th>person</th>
      <th>time</th>
      <th>days</th>
      <th>offer_ids</th>
      <th>amount</th>
      <th>reward</th>
      <th>became_member_on</th>
      <th>gender</th>
      <th>customer_id</th>
      <th>age_group</th>
      <th>income_range</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>7997</td>
      <td>0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-05-09</td>
      <td>1.0</td>
      <td>7997</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>10736</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-08-04</td>
      <td>NaN</td>
      <td>10736</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>15044</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2018-04-26</td>
      <td>2.0</td>
      <td>15044</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>9525</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-09-25</td>
      <td>NaN</td>
      <td>9525</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>6940</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-10-02</td>
      <td>NaN</td>
      <td>6940</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_merge.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event</th>
      <th>person</th>
      <th>time</th>
      <th>days</th>
      <th>offer_ids</th>
      <th>amount</th>
      <th>reward_x</th>
      <th>became_member_on</th>
      <th>gender</th>
      <th>customer_id</th>
      <th>...</th>
      <th>year</th>
      <th>difficulty</th>
      <th>duration_hours</th>
      <th>offer_id</th>
      <th>offer_type</th>
      <th>reward_y</th>
      <th>Web_channel</th>
      <th>Email_channel</th>
      <th>Social_channel</th>
      <th>Mobile_channel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>7997</td>
      <td>0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-05-09</td>
      <td>1.0</td>
      <td>7997</td>
      <td>...</td>
      <td>2017</td>
      <td>0.25</td>
      <td>168.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>10736</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-08-04</td>
      <td>NaN</td>
      <td>10736</td>
      <td>...</td>
      <td>2017</td>
      <td>1.00</td>
      <td>240.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>15044</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2018-04-26</td>
      <td>2.0</td>
      <td>15044</td>
      <td>...</td>
      <td>2018</td>
      <td>0.50</td>
      <td>168.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>9525</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-09-25</td>
      <td>NaN</td>
      <td>9525</td>
      <td>...</td>
      <td>2017</td>
      <td>0.50</td>
      <td>240.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>6940</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-10-02</td>
      <td>NaN</td>
      <td>6940</td>
      <td>...</td>
      <td>2017</td>
      <td>0.50</td>
      <td>120.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
final_merge.drop(['person','offer_id'], axis=1, inplace=True) #drop the repeated column

```


```python
final_merge.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event</th>
      <th>time</th>
      <th>days</th>
      <th>offer_ids</th>
      <th>amount</th>
      <th>reward_x</th>
      <th>became_member_on</th>
      <th>gender</th>
      <th>customer_id</th>
      <th>age_group</th>
      <th>income_range</th>
      <th>year</th>
      <th>difficulty</th>
      <th>duration_hours</th>
      <th>offer_type</th>
      <th>reward_y</th>
      <th>Web_channel</th>
      <th>Email_channel</th>
      <th>Social_channel</th>
      <th>Mobile_channel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-05-09</td>
      <td>1.0</td>
      <td>7997</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2017</td>
      <td>0.25</td>
      <td>168.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-08-04</td>
      <td>NaN</td>
      <td>10736</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
      <td>1.00</td>
      <td>240.0</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2018-04-26</td>
      <td>2.0</td>
      <td>15044</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2018</td>
      <td>0.50</td>
      <td>168.0</td>
      <td>2.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-09-25</td>
      <td>NaN</td>
      <td>9525</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
      <td>0.50</td>
      <td>240.0</td>
      <td>2.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-10-02</td>
      <td>NaN</td>
      <td>6940</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
      <td>0.50</td>
      <td>120.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_merge_cp = final_merge.copy()
```


```python
final_merge_cp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event</th>
      <th>time</th>
      <th>days</th>
      <th>offer_ids</th>
      <th>amount</th>
      <th>reward_x</th>
      <th>became_member_on</th>
      <th>gender</th>
      <th>customer_id</th>
      <th>age_group</th>
      <th>income_range</th>
      <th>year</th>
      <th>difficulty</th>
      <th>duration_hours</th>
      <th>offer_type</th>
      <th>reward_y</th>
      <th>Web_channel</th>
      <th>Email_channel</th>
      <th>Social_channel</th>
      <th>Mobile_channel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-05-09</td>
      <td>1.0</td>
      <td>7997</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2017</td>
      <td>0.25</td>
      <td>168.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-08-04</td>
      <td>NaN</td>
      <td>10736</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
      <td>1.00</td>
      <td>240.0</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2018-04-26</td>
      <td>2.0</td>
      <td>15044</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2018</td>
      <td>0.50</td>
      <td>168.0</td>
      <td>2.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-09-25</td>
      <td>NaN</td>
      <td>9525</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
      <td>0.50</td>
      <td>240.0</td>
      <td>2.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017-10-02</td>
      <td>NaN</td>
      <td>6940</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
      <td>0.50</td>
      <td>120.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_merge.shape
```




    (306534, 20)




```python

final_merge = final_merge.dropna(how='any',axis=0)
final_merge.shape
```




    (148805, 20)




```python
final_merge_cp = final_merge_cp.dropna(how='any',axis=0)
final_merge_cp.shape
```




    (148805, 20)




```python
# Data Analysis

#Q1: AGE segmentaion Split
#Q2: Income Segmentation Split
#Q3 Most Popular offer
#Q4 Gender SPlit for offertype
#Q5 Different Channel SPlit for offertype


```


```python
#Q1: AGE segmentaion Response:

```


```python
#Baby Boomers (Roughly 50 to 70 years old)
#Generation X (Roughly 35 – 50 years old)
#Millennials, or Generation Y (18 – 34 years old)
#Generation Z, or iGeneration (Teens & younger)

final_merge['age_group'] = final_merge['age_group'].map({1: 'Gen_Z', 2: 'Gen_Y', 3:'Gen_X', 4:'Baby_Boom'})

```


```python
final_merge['age_group'].value_counts()
```




    Gen_X        62727
    Gen_Y        38231
    Baby_Boom    26714
    Gen_Z        21133
    Name: age_group, dtype: int64




```python
plt.figure(figsize=(10,5))

final_merge.age_group.value_counts().reindex(['Gen_Z', 'Gen_Y', 'Gen_X', 'Baby_Boom']).plot(kind='bar', rot=0, figsize=(10,6));
plt.ylabel('Number of People');
plt.show();



labels=['Gen_Z', 'Gen_Y', 'Gen_X', 'Baby_Boom']
values=final_merge['age_group'].value_counts()

# Gen_X        62727
# Gen_Y        38231
# Baby_Boom    26714
# Gen_Z        21133

#% percentage is seen through Pie CHart 
explode=(0.10,0.10,0.10,0)

fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.show()
```


![png](output_68_0.png)



![png](output_68_1.png)



```python
#Insight1: It clearly shows max customer belong to Age- segment Gen_X which is 35-50.
```


```python
#Q1: Income segmentaion Response:
final_merge['income_range'] = final_merge['income_range'].map({1: 'Low', 2: 'Medium', 3:'high'})
```


```python
final_merge['income_range'].value_counts()
```




    Low       64095
    Medium    61655
    high      23055
    Name: income_range, dtype: int64




```python
plt.figure(figsize=(10,5))

final_merge.income_range.value_counts().reindex(['Low', 'Medium', 'high']).plot(kind='bar', rot=0, figsize=(10,6));
plt.ylabel('Number of People');

plt.show();

labels=['Low', 'Medium', 'high']
values=final_merge['income_range'].value_counts()

#Low       64095
#Medium    61655
#high      23055

#% percentage is seen through Pie CHart 
explode=(0.10,0.10,0.10)

fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.show()




```


![png](output_72_0.png)



![png](output_72_1.png)



```python
##Insight2: Low income customer base is more dominat.
```


```python
# Q3 Most Popular Offer

final_merge['offer_type'] = final_merge['offer_type'].map({1: 'BOGO', 2: 'Discount', 3: 'Informational'})

```


```python
final_merge['offer_type'].value_counts()
```




    BOGO             63834
    Discount         62311
    Informational    22660
    Name: offer_type, dtype: int64




```python
plt.figure(figsize=(10,5))

final_merge.offer_type.value_counts().reindex(['BOGO', 'Discount', 'Informational']).plot(kind='bar', rot=0, figsize=(10,6));
plt.ylabel('Number of People');

plt.show();

labels=['BOGO', 'Discount', 'Informational']
values=final_merge['offer_type'].value_counts()

#Low       64095
#Medium    61655
#high      23055

#% percentage is seen through Pie CHart 
explode=(0.10,0.10,0.10)

fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.show()


```


![png](output_76_0.png)



![png](output_76_1.png)



```python
#Insight3 Most Popular offer is BOGO followed by Discount.
```


```python
#Q4 Gender SPlit for offertype
```


```python
plt.figure(figsize=(9, 5))
g = sns.countplot(x='gender', hue="offer_type", data= final_merge[final_merge["gender"] != 3])
plt.title('Most Popular Offers to Each Gender')
plt.ylabel('Total')
plt.xlabel('Gender')
xlabels = ['Female', 'Male']
g.set_xticklabels(xlabels)
plt.legend(title='Offer Type')
plt.show();
```


![png](output_79_0.png)



```python
#Q5 Different Channel SPlit for offertype
```


```python
plt.figure(figsize=(9, 5))
g = sns.countplot(x='Web_channel', hue="offer_type", data= final_merge[final_merge["Web_channel"] !=  1])
plt.title('Most Popular Offers in Web_channel')
plt.ylabel('Total')
plt.xlabel('Web_channel')

plt.legend(title='Offer Type')
plt.show();

plt.figure(figsize=(9, 5))
g = sns.countplot(x='Email_channel', hue="offer_type", data= final_merge[final_merge["Email_channel"] != 2])
plt.title('Most Popular Offers in Email_channel')
plt.ylabel('Total')
plt.xlabel('Email_channel')

plt.legend(title='Offer Type')
plt.show();

plt.figure(figsize=(9, 5))
g = sns.countplot(x='Social_channel', hue="offer_type", data= final_merge[final_merge["Social_channel"] != 2])
plt.title('Most Popular Offers in Social_channel')
plt.ylabel('Total')
plt.xlabel('Social_channel')

plt.legend(title='Offer Type')
plt.show();


plt.figure(figsize=(9, 5))
g = sns.countplot(x='Email_channel', hue="offer_type", data= final_merge[final_merge["Mobile_channel"] != 2])
plt.title('Most Popular Offers in Mobile_channel')
plt.ylabel('Total')
plt.xlabel('Mobile_channel')
plt.legend(title='Offer Type')
plt.show();

```


![png](output_81_0.png)



![png](output_81_1.png)



![png](output_81_2.png)



![png](output_81_3.png)



```python
#Q6 Deepdive on Offer_type vs Event
final_merge['event'] = final_merge['event'].map({1: 'offer completed', 2: 'offer received', 3:'offer viewed',4: 'transaction'})                                            
```


```python
final_merge.event.value_counts()
```




    offer received     66501
    offer viewed       49860
    offer completed    32444
    Name: event, dtype: int64




```python


fig, ax = plt.subplots(figsize=(12,6))
sns.countplot(x="offer_type", hue="event", data=final_merge, palette="pastel")
plt.title("Total counts BOGO offer vs. 3-discount offer\n", fontsize=16)
final_merge['event'] = final_merge['event'].map({1: 'Completed', 2: 'Viewed'})





plt.figure(figsize=(10,5))
final_merge.event.value_counts().reindex(['Offer completed','Offer received', 'Offer viewed']).plot(kind='bar', rot=0, figsize=(10,6), color='tab:green');
plt.ylabel('Number of People');
plt.grid();final_merge['event'] = final_merge['event'].map({1: 'offer completed', 2: 'offer received', 3:'offer viewed'})


plt.figure(figsize=(10,5))
final_merge.event.value_counts().reindex(['Offer completed','Offer received', 'Offer viewed']).plot(kind='bar', rot=0, figsize=(10,6), color='tab:green');
plt.ylabel('Number of People');
plt.grid();
```


![png](output_84_0.png)



![png](output_84_1.png)



![png](output_84_2.png)



```python

#Data Modelling
```


```python
final_merge_cp.columns
```




    Index(['event', 'time', 'days', 'offer_ids', 'amount', 'reward_x',
           'became_member_on', 'gender', 'customer_id', 'age_group',
           'income_range', 'year', 'difficulty', 'duration_hours', 'offer_type',
           'reward_y', 'Web_channel', 'Email_channel', 'Social_channel',
           'Mobile_channel'],
          dtype='object')




```python
X = final_merge_cp[['offer_ids','amount','reward_x','difficulty','duration_hours','offer_type','gender','age_group','income_range']]
Y = final_merge_cp['event']

scaler = MinMaxScaler()

features = [ 'amount', 'reward_x', 'duration_hours']

X_scaled = X.copy()

X_scaled[features] = scaler.fit_transform(X_scaled[features])

X_scaled.head()
# creating training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)
```


```python
def predict_score(model):
    pred = model.predict(X_test)
    
    # Calculate the absolute errors
    errors = abs(pred - y_test)
    
    # Calculate mean absolute percentage error
    mean_APE = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mean_APE)
    
    return round(accuracy, 4)
```


```python
#decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)
print(f'Accuracy of Decision Tree classifier on training set are - {round(dtree.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy are - {predict_score(dtree)}%')
```

    Accuracy of Decision Tree classifier on training set are - 66.55%.
    Prediction Accuracy are - 87.8506%



```python
#KNN
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
print(f'Accuracy of K-NN classifier on training set are -  {round(knn.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy are -  {predict_score(knn)}%')
```

    Accuracy of K-NN classifier on training set are -  64.09%.
    Prediction Accuracy are -  85.3122%



```python
Conclusion:
    
    KNN and decision tree both have predication Accuracy almost similar. SO i would advise to use Decision Tree. 
```
