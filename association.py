# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:42:10 2023

@author: lenovo
"""



1]PROBLEM

::--ASSOCIATION RULES FOR 'book.csv' dataset




# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

#Loading The Dataset
book=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Association Rules/book.csv')
book

#EDA
book.head()
book.tail()
book.shape
book.describe()#Mathematical Calculations
book.isna().sum()#For NA values

# With 10% Support
frequent_itemsets=apriori(book,min_support=0.1,use_colnames=True)
frequent_itemsets

# with 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules

rules.sort_values('lift',ascending=False)

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]

# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


2. Association rules with 20% Support and 60% confidence
# With 20% Support
frequent_itemsets2=apriori(book,min_support=0.20,use_colnames=True)
frequent_itemsets2

# With 60% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2

# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

3. Association rules with 5% Support and 80% confidence
# With 5% Support
frequent_itemsets3=apriori(book,min_support=0.05,use_colnames=True)
frequent_itemsets3

# With 80% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
rules3

rules3[rules3.lift>1]

# visualization of obtained rule
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()




2]PROBLEM

::--ASSOCIATION RULES FOR 'my_movies.csv' dataset.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the Dataset
df = pd.read_csv("C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Association Rules/my_movies.csv")

#EDA
df.shape
df.head()
df.info()
df1 = df.iloc[:,5:]
df1
df.describe()
df.isna().sum()


# Apriori Algorithm
# 1. Association rules with 10% Support and 70% confidence

from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

# with 10% support
frequent_itemsets = apriori(df1,min_support = 0.1,use_colnames=True)
frequent_itemsets

# 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules

# Lift Ratio>1 is a good influential rule is selecting the associated
rules[rules.lift>1]

# Visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

# 2. Association rules with 5% Support and 90% confidence
# with 5% support
frequent_itemsets2=apriori(df1,min_support=0.05,use_colnames=True)
frequent_itemsets2

# 90% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.9)
rules2

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules2[rules2.lift>1]

# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()