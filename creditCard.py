#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[53]:


# loading the dataset to a Pandas DataFrame

credit_card_data = pd.read_csv(r"C:/Users/Alqama/Downloads/archive (2).zip")


# In[8]:


credit_card_data.head()


# In[9]:


credit_card_data.tail()


# In[10]:


# dataset information
credit_card_data.info()


# In[55]:


credit_card_data.describe()


# In[11]:


# checking missing values
credit_card_data.isnull().sum()


# In[12]:


# distribution of fair transaction and fraudulent transactions

credit_card_data['Class'].value_counts()


# In[13]:


# This dataset is very unbalanced where label 1 represents fraud transactions and 0 label represents fair transaction


# In[14]:


fair = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[15]:


print(fair.shape)
print(fraud.shape)


# In[57]:


# Bar plot for the number of fraudulent vs fair transcations
sns.countplot(x='Class', data=credit_card_data)
plt.title('Number of fraudulent vs fair transcations')
plt.show()


# In[58]:


# Creating fraudulent dataframe
data_fraud = credit_card_data[credit_card_data['Class'] == 1]
# Creating fair dataframe
data_non_fraud = credit_card_data[credit_card_data['Class'] == 0]


# In[79]:


# Distribution plot with KDE
plt.figure(figsize=(8, 5))
ax = sns.kdeplot(data_fraud['Time'], label='fraudulent', fill=True)
ax = sns.kdeplot(data_non_fraud['Time'], label='fair', fill=True)
ax.set(xlabel='Seconds elapsed between the transaction and the first transaction')
plt.legend()
plt.show()


# In[77]:


# Distribution plot of classes with amount
# KDE plot of classes with amount
plt.figure(figsize=(8, 5))
ax = sns.kdeplot(data_fraud['Amount'], label='fraudulent', fill=True)
ax = sns.kdeplot(data_non_fraud['Time'], label='fair', fill=True)
ax.set(xlabel='Transaction Amount')
plt.legend()
plt.show()


# In[ ]:


# Analysis
# We can see that the fraudulent transactions are mostly danced in the lower range of amount, whereas the non-fraudulent transactions are spread throughout low to high range of amount.


# In[66]:


# Standardization method
from sklearn.preprocessing import StandardScaler


# In[68]:


# Instantiate the Scaler
scaler = StandardScaler()


# In[69]:


# Fit the data into scaler and transform
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_train.head()


# In[70]:


# Checking the Skewness
# Listing the columns
cols = X_train.columns
cols


# In[73]:


# Plotting the distribution of the variables (skewness) of all the columns
k=0
plt.figure(figsize=(17,28))
for col in cols :    
    k=k+1
    plt.subplot(6, 5,k)    
    sns.histplot(X_train[col])
    plt.title(col+' '+str(X_train[col].skew()))


# In[80]:


# Mitigate skweness with PowerTransformer
# Importing PowerTransformer

from sklearn.preprocessing import PowerTransformer
# Instantiate the powertransformer
pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)
# Fit and transform the PT on training data
X_train[cols] = pt.fit_transform(X_train)


# In[81]:


# Transform the test set
X_test[cols] = pt.transform(X_test)


# In[82]:


# Plotting the distribution of the variables (skewness) of all the columns
k=0
plt.figure(figsize=(17,28))
for col in cols :    
    k=k+1
    plt.subplot(6, 5,k)    
    sns.histplot(X_train[col])
    plt.title(col+' '+str(X_train[col].skew()))


# In[67]:


# statistical measures of the data
fair.Amount.describe()


# In[17]:


fraud.Amount.describe()


# In[18]:


# Comparing the values for both transaction
credit_card_data.groupby('Class').mean()


# In[19]:


# building a sample dataset containing a similar distribution of normal transactions and fraud transaction
# fraud transactions --> 492


# In[20]:


fair_sample = fair.sample(n=492)


# In[21]:


new_dataset = pd.concat([fair_sample, fraud], axis=0)


# In[22]:


new_dataset.head()


# In[23]:


new_dataset.tail()


# In[24]:


new_dataset['Class'].value_counts()


# In[25]:


new_dataset.groupby('Class').mean()


# In[26]:


# splitting the data into Features & Targets 


# In[27]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[28]:


print(X)


# In[29]:


print(Y)


# In[30]:


# split the data into Training and Testing data
# 4 variables are required in this process


# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[32]:


print(X.shape, X_train.shape, X_test.shape)


# In[33]:


#Model training


# In[40]:


model = LogisticRegression()


# In[41]:


# Training the Logistic Regression model with training data
model.fit(X_train, Y_train)


# In[42]:


# Model evaluation
# Accuracy score


# In[43]:


# accuracy analysis on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[44]:


print('accuracy on training data: ', training_data_accuracy)


# In[110]:


# 1. Predicted probability
logistic_imb = LogisticRegression(C=0.01)
logistic_imb_model = logistic_imb.fit(X_train, Y_train)
Y_train_pred_proba = logistic_imb_model.predict_proba(X_train)[:, 1]


# In[111]:


# Plot the ROC curve
def draw_roc(actual, probs):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[112]:


# Call the function to draw the ROC curve
draw_roc(Y_train, Y_train_pred_proba)


# In[98]:


# Predicted probability
logistic_imb = LogisticRegression(C=0.01)
logistic_imb_model = logistic_imb.fit(X_train, Y_train)
Y_train_pred_proba = logistic_imb_model.predict_proba(X_train)[:, 1]


# In[99]:


# Convert predicted probabilities to binary class labels
threshold = 0.5
Y_train_pred = (Y_train_pred_proba >= threshold).astype(int)


# In[100]:


# Confusion matrix
confusion = metrics.confusion_matrix(Y_train, Y_train_pred)
print(confusion)


# In[101]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[106]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(Y_train, Y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[118]:


# ROC on the test set
# Predicted probability
Y_test_pred_proba = logistic_imb_model.predict_proba(X_test)[:, 1]


# In[119]:


# Plot the ROC curve
draw_roc(Y_test, Y_test_pred_proba)


# In[45]:


# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[46]:


print('accuracy on test data: ', test_data_accuracy)

