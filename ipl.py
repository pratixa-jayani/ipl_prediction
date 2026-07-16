#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#from ydata_profiling import ProfileReport


# In[2]:


# matches = pd.read_csv("matches.csv")
# deliveries = pd.read_csv("deliveries.csv")

# profile1 = ProfileReport(matches, explorative=True)
# profile1

# profile2 = ProfileReport(deliveries, explorative=True)
# profile2


# In[3]:


# profile1.to_file("matches_report.html")
# profile2.to_file("deliveries_report.html")


# # 1. Load Data

# In[4]:


matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

print(matches.shape)
print(deliveries.shape)


# In[6]:


matches.head()


# In[6]:


deliveries.head()


# # 2. EDA

# In[7]:


# Matches per season
matches['Season'].value_counts().sort_index().plot(kind='bar', title="Matches per Season")
plt.show()


# In[8]:


# Toss impact
toss_win = matches[matches['toss_winner'] == matches['winner']]
print("Toss win → match win %:",
      round((toss_win.shape[0] / matches.shape[0]) * 100, 2))



# In[9]:


# Top teams
matches['winner'].value_counts().head(8).plot(kind='bar', title="Top Winning Teams")
plt.show()


# In[10]:


# Toss impact
toss_win = matches[matches['toss_winner'] == matches['winner']]
print("Toss win → match win %:",
round((toss_win.shape[0] / matches.shape[0]) * 100, 2))


# In[11]:


# Score distribution
total_score = deliveries.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
first_innings = total_score[total_score['inning'] == 1]

sns.histplot(first_innings['total_runs'], kde=True)


# # 3. Preprocessing

# In[12]:


total_score = deliveries.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
total_score = total_score[total_score['inning'] == 1]

match_df = matches.merge(total_score[['match_id', 'total_runs']],
                         left_on='id', right_on='match_id')


# In[13]:


match_df.head()


# In[14]:


match_df['team1'].unique()


# In[15]:


teams=['Sunrisers Hyderabad',
       'Mumbai Indians', 
       'Royal Challengers Bangalore',
       'Kolkata Knight Riders',
       'Kings XI Punjab',
       'Chennai Super Kings',
       'Rajasthan Royals',
       'Delhi Capitals'
      ]


# In[16]:


# Fix team names
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')


# In[17]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[18]:


match_df = match_df[match_df['dl_applied'] == 0]


# In[19]:


match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]


# In[20]:


delivery_df = match_df.merge(deliveries, on='match_id')


# In[21]:


delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[22]:


delivery_df.shape


# In[23]:


delivery_df.head()


# # 4. Feature Engineering

# In[24]:


delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()


# In[25]:


delivery_df.head()


# In[26]:


delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']


# In[27]:


delivery_df.head()


# In[28]:


delivery_df['balls_left'] = 120 - ((delivery_df['over'] - 1)*6 + delivery_df['ball'])


# In[29]:


delivery_df.head()


# In[30]:


delivery_df['player_dismissed'] = delivery_df['dismissal_kind'].notna().astype(int)
delivery_df['wickets'] = 10 - delivery_df.groupby('match_id')['player_dismissed'].cumsum()


# In[31]:


delivery_df.head()


# In[32]:


delivery_df.tail()


# In[33]:


delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']


# In[34]:


delivery_df.head()


# In[35]:


delivery_df['result'] = (delivery_df['batting_team'] == delivery_df['winner']).astype(int)
delivery_df.head()


# In[36]:


final_df = delivery_df[['batting_team', 'bowling_team', 'city',
                        'runs_left', 'balls_left', 'wickets',
                        'total_runs_x', 'crr', 'rrr', 'result']]


# In[37]:


final_df.head()


# In[38]:


final_df = final_df.dropna()
final_df = final_df[final_df['balls_left'] != 0]


# In[39]:


final_df.head()


# # 5. Model Training
# 

# In[40]:


X = final_df.drop('result', axis=1)
y = final_df['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)


# In[41]:


X_train.head()


# In[42]:


trf = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), ['batting_team', 'bowling_team', 'city'])], remainder='passthrough')


# # 6.Logistic Regression

# In[43]:


pipe_lr = Pipeline([
    ('preprocessor', trf),
    ('model', LogisticRegression(solver='liblinear'))
])

pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", acc_lr)


# # 7. Random Forest

# In[44]:


pipe_rf = Pipeline([
    ('preprocessor', trf),
    ('model', RandomForestClassifier(n_estimators=100, random_state=1))
])

pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", acc_rf)


# In[45]:


pipe_lr = Pipeline([
    ('preprocessor', trf),
    ('model', LogisticRegression(max_iter=1000))
])


# In[46]:


pipe_lr.fit(X_train, y_train)


# In[47]:


y_pred = pipe_lr.predict(X_test)


# In[48]:


y_pred


# # 8. Best Model + Confusion Matrix

# In[49]:


best_model = pipe_lr
y_pred_best = y_pred

print("Selected Model: Logistic Regression")

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# # 9. Save Model

# In[50]:


pickle.dump(best_model, open('pipe.pkl', 'wb'))
print("Model saved")


# In[ ]:




