import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
df = pd.read_excel('./archive/top5-players.xlsx')

# brief looking of the datasets
print(df.info)
print(df.isnull().sum())

# stats check
des = df.describe()
print(des)

# mean avg goals
average_goals = df['Gls'].mean()
average_assists = df['Ast'].mean()
print(f"Average Goals: {average_goals}")
print(f"Average Assists: {average_assists}")

# top goalscorers
top_scorers = df.nlargest(5,['Gls'])
print(top_scorers[['Player','Gls']])

# visualize datasets in general 
df_numeric = df.select_dtypes(include=['float64', 'int64'])
df_numeric.hist(figsize=(16, 12), bins=30, color='blue', edgecolor='black')
plt.suptitle('Histograms of All Numeric Columns', fontsize=20)
plt.show()

# visualize datasets
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['xG', 'Gls'])
plt.figure(figsize=[10,6])
plt.scatter(df['xG'],df['Gls'],color = 'blue',alpha=0.5)
a, b = np.polyfit(df['xG'], df['Gls'], 1)
plt.plot(df['xG'], a * df['xG'] + b, color='red')
plt.title('Compare xG to real Goals')
plt.xlabel('xG')
plt.ylabel('Goals')
plt.grid(True)
plt.show()

# goal distribution 
plt.figure(figsize=(10, 6))
sns.histplot(df['Gls'], kde=True, color='blue')
plt.title('Gls distribution')
plt.xlabel('Goals')
plt.ylabel('Frequency')
plt.show()

# features and label
features = ['Age', 'MP', 'Starts', 'Min', '90s', 'xG', 'npxG', 'xAG', 
            'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls_90', 'Ast_90', 
            'G+A_90', 'G-PK_90', 'G+A-PK_90', 'xG_90', 'xAG_90', 'xG+xAG_90', 
            'npxG_90', 'npxG+xAG_90']

target = 'Gls'

# data handling and cleaning
df = pd.get_dummies(df,columns=['Pos'],drop_first=True)
num_cols = df.select_dtypes(include=['float64','int64']).columns
string_cols = df.select_dtypes(include=['object']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
df[string_cols] = df[string_cols].fillna('Unknown')

# split data
X = df[features]
Y = df[target]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# train data using linear regression
model = LinearRegression()
model.fit(X_train,Y_train)

# predict
Y_pred = model.predict(X_test)

# get some stats about the results
mae = mean_absolute_error(Y_test,Y_pred)
mse = mean_squared_error(Y_test,Y_pred)
r2 = r2_score(Y_test,Y_pred)
print(f'MAE: {mae}, MSE: {mse}, R2: {r2}')