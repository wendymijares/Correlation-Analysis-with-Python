#!/usr/bin/env python
# coding: utf-8

# ## Importing library

# In[51]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) #Adjust the configuration the plot we will create

pd.options.mode.chained_assignment = None

# Now we need to read in the data
df = pd.read_csv(r"C:\Private wen\Courses\DA YOUTUBE 1\Porfolio\movies.csv")


# In[21]:


## Look the data
df.head()


# In[23]:


df.tail()


# In[22]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


# We need to see if we have any missing data
# Let's loop through the data and see if there is anything missing

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[9]:


# Data Types for our columns

print(df.dtypes)


# In[61]:


## Remove all duplicates
df = df.drop_duplicates()
df


# In[83]:


## Change  Null values for 0
df = df.fillna(0)
df


# In[ ]:





# In[85]:


## Change a data type in columns

df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')
df


# In[24]:


##Create a new column to correct the released year column
df['year_correct'] = df['released'].astype(str).str[:4]
df


# In[34]:


##Oder by the column Descending
df =df.sort_values (by=['gross'], inplace=False , ascending=False)


# In[26]:


## To see all the data
pd.set_option('display.max_rows',None)


# In[29]:


##Drop any duplicates
df['company'].drop_duplicates().sort_values(ascending=False)


# In[39]:


## Scatter plot Budget vs gross

plt.scatter(x=df['budget'], y=df['gross'])

## Title
plt.title('Budget vs Gross Earnings')

##Labels
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for film')

plt.show()


# In[36]:


df.head()


# In[40]:


## Define if is exist any correlation using SEABORM REGPLOT

sns.regplot(x="gross", y="budget", data=df)


# In[42]:


##Changin the color

sns.regplot(x="gross", y="budget", data=df, scatter_kws={"color":"green"},line_kws={"color":"blue"})


# In[43]:


## Correlation
## Correlation types 
## PEARSAN (by default) , KENDALL, SPEARMAN
df.corr()


# In[45]:


df.corr(method ='pearson')


# In[46]:


df.corr(method ='kendall')


# In[47]:


df.corr(method ='spearman')


# In[48]:


#Visualize 

correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[49]:


## Numerize to change a string colum into a numeric

df_numerized = df


for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name]= df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[59]:


df.head()


# In[53]:


### Corr 
correlation_matrix = df_numerized.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[54]:


df_numerized.corr()


# In[55]:


correlation_mat = df_numerized.corr()

corr_pairs = correlation_mat.unstack()

corr_pairs


# In[56]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[58]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]
high_corr


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




