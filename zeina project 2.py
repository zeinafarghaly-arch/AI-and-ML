#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1)Data Load

# In[2]:


df= pd.read_csv(r"C:\Users\Lenovo\Downloads\train(1).csv")
df.head()


# In[3]:


df.describe()


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.isna().sum()


# In[7]:


df.info()


# In[8]:


y = df["Survived"]
X = df.drop("Survived", axis=1)


# 2)preprocessing

# In[9]:


X = X.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)


# In[10]:


X["Age"] = X["Age"].fillna(X["Age"].median())


# In[11]:


X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])


# In[12]:


X = pd.get_dummies(X, columns=["Sex","Embarked"], drop_first=True)


# 3)train

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# 4)models

# In[17]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_lr= regressor.predict(X_test)


# In[18]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# 5)evaluation

# In[23]:


y_test = y_test.astype(int)
y_pred_lr = y_pred_lr.astype(int)
y_pred_svm = y_pred_svm.astype(int)
y_pred_knn = y_pred_knn.astype(int)


# In[24]:


from sklearn.metrics import classification_report, confusion_matrix
print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr), "\n")

print("SVM:")
print(classification_report(y_test, y_pred_svm, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm), "\n")

print("KNN:")
print(classification_report(y_test, y_pred_knn, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


# In[25]:


from sklearn.metrics import confusion_matrix

def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion(y_test, y_pred_lr, "Logistic Regression")
plot_confusion(y_test, y_pred_svm, "SVM")
plot_confusion(y_test, y_pred_knn, "KNN")


# In[ ]:




