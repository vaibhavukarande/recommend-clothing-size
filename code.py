# --------------
# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Code starts here
df=pd.read_json(path,lines=True)
#print(df.head())

df.columns=df.columns.str.strip().str.replace(" ","_")
print(df.columns)

mising_data=pd.DataFrame({"Total Missing": df.isnull().sum(),"Percentage_Missing": df.isna().sum()*100/df.shape[0]})
#print(mising_data)

df.drop(['waist', 'bust', 'user_name','review_text','review_summary','shoe_size','shoe_width'],axis=1,inplace=True)

print(df.columns)

#Store all the features(independent values) in a variable called X
X=df.drop(['fit'],axis=1)

#Store the target variable fit(dependent value) in a variable called y
y=df['fit']

#Split the dataframe 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=6)

# Code ends here


# --------------
def plot_barh(df,col, cmap = None, stacked=False, norm = None):
    df.plot(kind='barh', colormap=cmap, stacked=stacked)
    fig = plt.gcf()
    fig.set_size_inches(24,12)
    plt.title("Category vs {}-feedback -  cloth {}".format(col, '(Normalized)' if norm else ''), fontsize= 20)
    plt.ylabel('Category', fontsize = 18)
    plot = plt.xlabel('Frequency', fontsize=18)


# Code starts here
g_by_category=df.groupby(['category'])
print(g_by_category)
cat_fit=g_by_category['fit'].value_counts().unstack()
print(cat_fit)

cat_fit.plot.bar()
# Code ends here


# --------------
# Code starts here



# Check the value counts of g_by_category based on length
cat_len=g_by_category['length'].value_counts().unstack()
print(cat_len)

# plot the bar chart for cat_len
cat_len.plot.barh()

# Code ends here


# --------------

# Code starts here
#Create the function get_cms pass the parameter x
def get_cms(x):
    if type(x)==float:
        return x
    try:
        return (int(x[0])*30.48) + (int(x[4:-2])*2.54)
    except:
        return(int(x[0])*30.48)
   
X_train.height=X_train.height.apply(get_cms)
X_test.height=X_test.height.apply(get_cms)

# Code ends here


# --------------
# Code starts here



#Check the missing values of X_train

print(X_train.isnull().sum())

#Drop the rows from columns ['height','length','quality'] which contains the NaN values 
#Drop the rows from columns ['height','length','quality'] which contains the NaN values

X_train.dropna(subset=['height','length','quality'],inplace=True)
X_test.dropna(subset=['height','length','quality'],inplace=True)

#Drop the index (records) from y_train which is not contained X_train
#Drop the index (records) from y_test which is not contained in X_test
y_train=y_train[X_train.index]
y_test=y_test[y_test.index]

#Fill the missing values for columns bra_size and hips with mean on X_train
#Fill the missing values for columns bra_size and hips with mean on X_test
X_train['bra_size'].fillna(X_train['bra_size'].mean(),inplace=True)
X_test['bra_size'].fillna(X_train['bra_size'].mean(),inplace=True)
print(X_train.isnull().sum())
print(X_test.isnull().sum())

#Calculate the mode of theX_train using X_train['cup_size'].mode()[0] 
#Calculate the mode of the X_test usingX_test['cup_size'].mode()[0] 

mode_1=X_train['cup_size'].mode()[0]
mode_2=X_test['cup_size'].mode()[0]

#Replace the Nan values with the mode_1 on X_train['cup_size']
#Replace the Nan values with the mode_2 on X_test['cup_size']

X_train['cup_size']=X_train['cup_size'].replace(np.nan,mode_1)
X_test['cup_size']=X_test['cup_size'].replace(np.nan,mode_2)
print(X_train.isnull().sum())
print(X_test.isnull().sum())
# Code ends here


# --------------
# Code starts here




X_train=pd.get_dummies(X_train,columns=['category', 'cup_size','length'],prefix=['category', 'cup_size','length'])

X_test=pd.get_dummies(X_test,columns=['category', 'cup_size','length'],prefix=['category', 'cup_size','length'])
# Code ends here


# --------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score




# code starts here 

# Instantiate logistic regression
model = DecisionTreeClassifier(random_state = 6)

# fit the model
model.fit(X_train,y_train)

# predict the result
y_pred =model.predict(X_test)

# calculate the f1 score
score = accuracy_score(y_test, y_pred)
print(score)

# calculate the precision score
precision = precision_score(y_test, y_pred, average=None)
print(precision)






# --------------
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# parameters for grid search
parameters = {'max_depth':[5,10],'criterion':['gini','entropy'],'min_samples_leaf':[0.5,1]}

# Code starts here
model=DecisionTreeClassifier(random_state=6)

grid=GridSearchCV(estimator=model,param_grid=parameters)

grid.fit(X_train,y_train)

y_pred=grid.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)


# Code ends here


