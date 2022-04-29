#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Libraries:
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# The multivariate normal, multinormal or Gaussian distribution is a generalization of the one-dimensional normal distribution to higher dimensions. Such a distribution is specified by its mean and
# covariance matrix. These parameters are analogous to the mean (average or "center") and variance (standard deviation, or "width," squared) of the one-dimensional normal distribution.

# # 
# 
# Read the files labeled as 'sampleX.txt' using numpy or pandas and plot them.

#  # Read the files:

# In[25]:


link_address = ["sample1.txt" , "sample2.txt" , "sample3.txt"]
which_file = ["file1" , "file2" , "file3"]
file_name = ["sample1:" , "sample2:" , "sample3:"]
file_list = []
for i in range(3):
    file = pd.read_csv(link_address[i] ,delimiter="\t")
    print(file_name[i])
    print(file.head())
    file_list.append(file)
    print(" "),

#now I wanna put each file in a variable to call that later...
file1 = file_list[0]
file2 = file_list[1]
file3 = file_list[2]


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15, 3))

#lists that I need:
files = [file1 , file2 , file3]
wanted_colors = ["mediumblue" , "crimson" , "forestgreen"]
fig_names = ["Sample1" , "Sample2" , "Sample3"]

for i in range(3):
    ax = plt.subplot(1,3,i+ 1)
    x = files[i]["x"]
    y = files[i]["y"]
    fig = plt.scatter(x , y , marker = "." , color = wanted_colors[i])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(fig_names[i])
    
plt.show()


# # Correlation

# 
# Determine whether each sample is correlated, anticorrelated or uncorrelated.

# In[27]:


files = [file1 , file2 , file3]
for i in range(3):
    wanted = np.corrcoef(files[i].x , files[i].y)
    print("the correlation_coefficient of sample" , i , "is" , wanted[0,1])


# So the Sample 1 is "correlated" Sample 2 in "anticorrelated" & the last one is "uncorrelated"...

# ## 3d Plot
# Plot the joint probability distribution of each sample in 3D. For this you can use 'plot_surface' found in matplotlib library.

# If you want to plot using matplotlib, the codes below will come in handy. First line makes sure that your plots are interactive, second line provides color maps.

# In[28]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import cm


# At the first step try to get the 2D histogram of your data. (Hint: beware of different sizes of arrays!)

# In[29]:


#Histograms:
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15, 3))
for i in range(3):
    ax = plt.subplot(1,3,i+ 1)
    x = files[i]["x"]
    y = files[i]["y"]
    plt.hist2d(x, y , bins = 100)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(fig_names[i])
    
plt.show()


# Now you can plot the 3D histogram:

# In[31]:


#Plot Sample 1:

import matplotlib
matplotlib.rc_file_defaults()

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')

#Make data suitable to 3d plotting:
z1 , oldx1 , oldy1 = np.histogram2d(files[0]["x"] , files[0]["y"], bins =100)
x = np.zeros(100)
for i in range(0,100):
  x[i] = (oldx1[i] + oldx1[i+1])/2

y = np.zeros(100)
for i in range(0,100):
  y[i] = (oldy1[i] + oldy1[i+1])/2

x, y = np.meshgrid(x , y)

#Plot the surface.
surf = ax.plot_surface(x, y, z1 , cmap= "GnBu" , linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.title("Sample 1")
plt.show()


# In[32]:


#Plot Sample 2:

import matplotlib
matplotlib.rc_file_defaults()
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')

#Make data suitable to 3d plotting:
z2 , oldx2, oldy2 = np.histogram2d(files[1]["x"] , files[1]["y"] , bins =100)

x = np.zeros(100)
for i in range(0,100):
  x[i] = (oldx2[i] + oldx2[i+1])/2

y = np.zeros(100)
for i in range(0,100):
  y[i] = (oldy2[i] + oldy2[i+1])/2

x, y = np.meshgrid(x , y)

#Plot the surface.
surf = ax.plot_surface(x, y, z2 , cmap= "Oranges" , linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.title("Sample 2")
plt.show()


# In[33]:


#Plot Sample 3:

import matplotlib
matplotlib.rc_file_defaults()
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')

#Make data suitable to 3d plotting:
z3 , oldx3, oldy3 = np.histogram2d(files[2]["x"] , files[2]["y"] , bins =100)

x = np.zeros(100)
for i in range(0,100):
  x[i] = (oldx3[i] + oldx3[i+1])/2

y = np.zeros(100)
for i in range(0,100):
  y[i] = (oldy3[i] + oldy3[i+1])/2

x, y = np.meshgrid(x , y)

#Plot the surface.
surf = ax.plot_surface(x, y, z3 , cmap= "Reds" , linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.title("Sample 3")
plt.show()


# I tried to code in a more compact way like the below cell but the output was not the wanted result...

# import matplotlib
# matplotlib.rc_file_defaults()
# 
# fig = plt.figure(figsize=(6, 6))
# ax = plt.axes(projection='3d')
# 
# #Make data suitable to 3d plotting
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     z , oldx , oldy = np.histogram2d(files[i]["x"] , files[i]["y"] , bins =100)
#     
#     print(oldx.shape)
#     print(oldx[22])
#     x = np.zeros(100)
#     for j in range(0,100):
#         x[i] = (oldx[j] + oldx[j+1])/2
# 
#     y = np.zeros(100)
#     for k in range(0,100):
#         y[i] = (oldy[k] + oldy[k+1])/2
#     
#     print(x.shape)
#     
#     x , y = np.meshgrid(x , y)
#     #print(m.shape)
#     #print(m)
#     #print(n)
#     #Plot the surface.
#     surf = ax.plot_surface(x, y, z , cmap= "GnBu" , linewidth=0, antialiased=False)
# 
#     # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=10 , ax = plt.axes(projection='3d'))
#     plt.title("Sample 1")
#     
# plt.show()

# 
# Using the calculated histograms, now write a code to calculate the marginalized PDFs along both axes and then plot them.

# In[34]:


#Calculate marginalized distributions :

#file1:
marginal1_x = np.sum(z1 , axis = 1)
marginal1_y = np.sum(z1 , axis = 0)
#print(marginal1_x)
#print(type(marginal1_x))
#print(marginal1_x.shape)
u1 = np.delete(oldx1 , -1)
v1 =np.delete(oldy1 , -1)

#file2:
marginal2_x = np.sum(z2 , axis = 1)
marginal2_y = np.sum(z2 , axis = 0)
u2 = np.delete(oldx2 , -1)
v2 =np.delete(oldy2 , -1)

#file3:
marginal3_x = np.sum(z3 , axis = 1)
marginal3_y = np.sum(z3 , axis = 0)
u3 = np.delete(oldx3 , -1)
v3 =np.delete(oldy3 , -1)


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set() #Use this line to plot the marginalized PDFs with seaborn style

plt.figure(figsize=(15, 10))
plt.suptitle('Marginalized PDFs')

plt.subplot(321)
plt.bar(u1 , marginal1_x ,width = 0.1 , color ="deeppink")
plt.xlabel('y_sample1')
plt.ylabel('Marginalized PDF(x)')

plt.subplot(322)
plt.bar(v1 , marginal1_y ,width = 0.1 , color ="cyan")
plt.xlabel('x_sample1')
plt.ylabel('Marginalized PDF(y)')

plt.subplot(323)
plt.bar(u2 , marginal2_x ,width = 0.1 , color ="deeppink")
plt.xlabel('y_sample2')
plt.ylabel('Marginalized PDF(x)')

plt.subplot(324)
plt.bar(v2 , marginal2_y ,width = 0.1 , color ="cyan")
plt.xlabel('x_sample2')
plt.ylabel('Marginalized PDF(y)')

plt.subplot(325)
plt.bar(u3 , marginal3_x ,width = 0.1 , color ="deeppink")
plt.xlabel('y_sample3')
plt.ylabel('Marginalized PDF(x)')

plt.subplot(326)
plt.bar(v3 , marginal3_y ,width = 0.1 , color ="cyan")
plt.xlabel('x_sample3')
plt.ylabel('Marginalized PDF(y)')

plt.show()


# ## Extra example:
# 
# You can also combine the two steps above and plot the joint PDF and the marginalized ones altogether using seaborn.

# In[36]:


g1 = sns.jointplot(data=file1, x='x', y='y', kind='hist')
g1.fig.suptitle('Sample 1')
g1.fig.tight_layout()

g2 = sns.jointplot(data=file2, x='x', y='y', kind='hist')
g2.fig.suptitle('Sample 2')
g2.fig.tight_layout()

g3 = sns.jointplot(data=file3, x='x', y='y', kind='hist')
g3.fig.suptitle('Sample 3')
g3.fig.tight_layout()

plt.show()


# ## Contour Plots

# Plot the contours of the datasets showing different values of contours.

# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rc_file_defaults() #Use this line to revert back to matplotlib default style
#Plot the contours:
plt.figure(figsize=(15, 3))
#Sample1:
plt.subplot(131)
plt.contour(x, y, z1, 5, cmap = 'RdGy')
plt.title("Sample1")
#Sample2:
plt.subplot(132)
plt.contour(x, y, z2, 5, cmap = 'RdGy')
plt.title("Sample2")
#Sample3:
plt.subplot(133)
plt.contour(x, y, z3, 5, cmap = 'RdGy')
plt.title("Sample3")


# 
# ## 3 parts
# In the multivariate case, a gaussian distribution is defined via a mean and a covrience matrix. Here the covarience matrix is the equivalant of varience in higher dimensions. To refresh your mind, take a look at the [Wikipedia page](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Notation_and_parameterization). A correlation matrix is very similiar but has some [subtle differences](https://en.wikipedia.org/wiki/Correlation#Correlation_matrices). 

# Now using those defenitions, find the **covariance** (part 1) and **correlation** (part 2) matrices for each distribution. Are they the same? why? (part 3)
# 
# (Hint 1: You may find scipy.stats useful)
# 
# (Hint 2: Study the lecture note titled: 'parameter estimation 7' once more)
# 
# (Hint 3: [This lecture note](https://pages.ucsd.edu/~rlevy/lign251/fall2007/lecture_4.pdf) may also be useful, althogh the codes aren't written in python)

# In[42]:


files = [file1 , file2 , file3]
fig_names = ["Sample1" , "Sample2" , "Sample3"]
for i in range(3):
    covariance_matrix = files[i].cov()
    correlation_matrix = files[i].corr()
    print("Covariance matrix of" ,fig_names[i] ,"is :")
    print(covariance_matrix)
    print("  ")
    print("Correlation matrix of" ,fig_names[i] ,"is :")
    print(correlation_matrix)


# 
# ## 2 parts
# 
# Now, only focus on the positievly correlated distribution. If the errors along both of the axes are huge, (as discussed in the lecture 'parameter estimation 7'), Is there a linear combination of the two parameters that can be well constrained? Discuss it (part 1).  Find the mode of the distribution (part 2)

# In[15]:





# # Real World
# let's apply this to real world data and using house price data. first import house_data.csv

# In[16]:


df = pd.read_csv("House_price.csv")
df.head(8)


# you can see detail of your dataframe with the code below

# In[17]:


df.info()


# Now select the columns of the train set with numerical data

# In[18]:


df_num = df.select_dtypes(include='number')
df_num


# In[19]:


df_num["LotArea"]


# Plot the distribution of all the numerical data

# In[43]:


column_list = []
for column in df_num.columns:
    column_list.append(column)
print(column_list)

get_ipython().run_line_magic('matplotlib', 'inline')
for col in column_list:
    var = df_num[col]
    plt.plot(var)
    plt.show()


# plot Heatmap for all the remaining numerical data including the 'SalePrice'

# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')
for col in column_list:
    var = df_num[col]
    price = df_num["SalePrice"]
    plt.hist2d(var , price , bins =100)
    plt.show()


# From the distribution of each numerical variables as well as the heatmap you can notice columns that are important and correlated (correlation higher than absolute 0.3) with our target variable 'SalePrice'. select columns where the correlation with 'SalePrice' is higher than |0.3|
# 

# In[ ]:


#Code here


# Now choose Features with high correlation (higher than 0.5) and plot the correlation of each feature with SalePrice

# In[ ]:


#code here


# Check the NaN of dataframe set by ploting percent of missing values per column and plot the result

# In[ ]:


#Code here


# in the last session I think Amirreza said that droping Nan cells is not suited in many projects cause
# it can remove alots of information of your dataframe. ofcourse he is right and I would like to give a short introduction to the process of handling Nan cells which is called "Imputation". Data imputation is the substitution of estimated values for missing or inconsistent data items (fields). The substituted values are intended to create a data record that does not fail edits. here you can use Simple_

# In[ ]:


# Imputation of missing values (NaNs) with SimpleImputer you can check diffrent strategy 
my_imputer = SimpleImputer(strategy="median")
df_num_imputed = pd.DataFrame(my_imputer.fit_transform(df_num))
df_train_imputed.columns = df_train_num.columns


# # Categorical features

# ## Explore and clean Categorical features

# find all Catagorical columns. you can use the code for finding the numerical columns and just using 'object' for dtype.

# In[46]:


df_cat = df.select_dtypes(include='object')
df_cat


# Countplot for each of the categorical features in the train set

# In[62]:


cat_list = []
for cat in df_cat.columns:
    cat_list.append(cat)
print(cat_list)

get_ipython().run_line_magic('matplotlib', 'inline')
for c in cat_list:
    var = df_cat[c]
    sns.countplot(var)
    plt.show()


# In[ ]:




