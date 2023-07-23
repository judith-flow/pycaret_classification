import pycaret
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime


#####DATA EXPLORATION


# READ DATA

files = glob.glob(os.path.join("data/*.csv"))

df_list = []


for filename in files:
    df = pd.read_csv(filename, index_col=None, header=0,dtype=datatype,parse_dates=['sign_up_time'])
    df_list.append(df)

# merge multiple csv files 
df = pd.concat(df_list, axis=0, ignore_index=True)

# Save dataframe into a new csv file

df.to_csv('df_ios_time.csv', index=False)




# view the dataframe
df.head()
df.info()
df.describe()

# check duplicate of user_id
df.user_id.unique().shape[0]


# check NULL values

max_rows = df['user_id'].count()
print('% Missing Data:')
print((1 - df.count() / max_rows) * 100)

# delete columns with high missing value percentage
df.drop(['first_export_time_ios','first_corners_time_ios'],axis=1, inplace=True)



# convert time columns into datetime datatype
df['sign_up_time'] = pd.to_datetime(df['sign_up_time'], errors='coerce')
df['first_project_time_ios'] = pd.to_datetime(df['first_project_time_ios'], errors='coerce')
df['first_room_time_ios'] = pd.to_datetime(df['first_room_time_ios'], errors='coerce')
df['first_camera_time_ios'] = pd.to_datetime(df['first_camera_time_ios'], errors='coerce')
df['first_square_time_ios'] = pd.to_datetime(df['first_square_time_ios'], errors='coerce')


# check the distribution of feature: combine value_counts and percentage
# 
vc = df['converted'].value_counts().to_frame().reset_index()
vc['percent'] = vc["converted"].apply( lambda x : round(100*float(x)/ len(df), 2))
vc = vc.rename(columns = {"index" : "converted" , "converted" : "count"})
vc


# check country distribution of users
vc = df['country'].value_counts().to_frame().reset_index()
vc['percent'] = vc["country"].apply( lambda x : round(100*float(x)/ len(df), 2))
vc = vc.rename(columns = {"index" : "country" , "country" : "count"})
vc

#check the time range of sign up time of users
#check if the range is abnormal
print(df['sign_up_time'].min(),"-",df['sign_up_time'].max())


# group by target feature and device type
df.groupby(['converted','first_device_type']).agg({'user_id':'count'})




# clean unreasonable data: the time difference shouldn't be smaller than 0
time_diff = ['first_project_time_diff','first_room_time_diff']
df = df[df['first_project_time_diff']>0]




#######DATA VISUALISATION 

# print all categorical features bar chart 
import matplotlib.gridspec as gridspec

cat_features = ["marketing_use_case","marketing_intent","industry_claim","industry_other","industry_reno","industry_surveys","marketing_team","first_device_type","first_device_lidar","agg_first_room_create_method"]

df_1 = df.loc[df['converted'] == '1'].reset_index()
df_0 = df.loc[df['converted'] == '0'].reset_index()
df_1 = df_1.drop('index', axis=1)
df_0 = df_0.drop('index', axis=1)

def visualize_split(cat_feature):
    gs = gridspec.GridSpec(1, 2) 
    fig = plt.figure(figsize=(26,16))
    #Using the 1st row and 1st column for plotting 
    ax = plt.subplot(gs[0,0])
    ax = sns.countplot(y=cat_feature, hue='converted',data=df_1 ,palette=['#432371']) 
    total = len(df_1[cat_feature])
    for p in ax.patches:
        percentage = '{:.4f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
    #Using the 1st row and 2nd column for plotting 
    ax1=plt.subplot(gs[0,1])
    ax1=sns.countplot(y=cat_feature, hue='converted',data=df_0 )
    total = len(df_0[cat_feature])
    for p in ax1.patches:
        percentage = '{:.4f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax1.annotate(percentage, (x, y))
    plt.show()
    
for i in range(len(cat_features)):
    visualize_split(cat_features[i])
    

# Pair plot
sns.pairplot(df, hue="converted")


# correlation matrix
# adjust figsize to show everything 
corr_matrix = df.corr()
plt.figure(figsize = (16,8))
sns.heatmap(corr_matrix, annot=True, linewidths=.5)
plt.show()


# densitiy visualisation
sns.displot(df_unseen_balanced, x="room_method_camera_ios",hue='converted', kind = "kde")






#######MODEL TRAINING

# prepare the training dataset
# the convertion is too unbalanced
df_1 = df.loc[df['converted'] == '1'].reset_index()
df_0 = df.loc[df['converted'] == '0'].reset_index()
df_1 = df_1.drop('index', axis=1)
df_0 = df_0.drop('index', axis=1)
    
class_count_0, class_count_1 = df['converted'].value_counts()


#undersampling unconverted users to create a training data set
df_0_sampled = df_0.sample(class_count_1)

df_balanced = pd.concat([df_1, df_0_sampled], ignore_index=True, sort=False)

df_balanced.info()



# train test data split 
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_balanced, test_size=0.2)
print(train.shape)
print(test.shape)



# Pycaret AutoML classification 

'''Hyperparameters in setup function:

1. outliers_threshold
2. feature_selection_threshold
3. fold
4. normalize_method
5. categorical_imputation
6. numeric_imputation'''

s = setup(data= train, 
          target='converted',
          normalize=True,         
          remove_outliers=True, 
 
          silent=True,
          normalize_method='robust',
          categorical_imputation='constant',
          numeric_imputation='mean',
          ignore_features=['user_id'], #key 
          high_cardinality_features=['country','first_device_model'],
          numeric_features= num_features,
          categorical_features=cat_features,
          combine_rare_levels = True,
          ignore_low_variance = True,              
          outliers_threshold = 0.01,
          remove_multicollinearity=True,
          feature_selection = True,
          feature_selection_threshold = 0.85,
          fold = 5,  # cross validation
          n_jobs = -1,
          session_id=123)


# show the evalution of best models 
best_model = compare_models()


# we prioritize "recall" 
algos = ['lightgbm', 'rf','gbc', 'ada'] # models with recall > %70
best_choosen = compare_models(include = algos, n_select = len(algos), sort = 'recall')
# save the best model method
best_model = best_choosen[0]


# create a model based on the best model method
m = create_model(best_model)





####### TUNING AND EVALUATION

# ensemble the model and cross validation, optimizing "recall"
bagged_model = ensemble_model(best_model, fold = 10, optimize = 'recall')

# search through scikit learn library to tune the model 
tuned_model_scikit = tune_model(best_model,
                      n_iter = 20,
                      optimize='recall', #the best model is selected based on the recall metric
                      search_library='scikit-learn',
                      custom_grid = params,
                      #return_tuner=True,
                      choose_better=True)



# use "optuna" library to tune the model
tuned_model_optuna = tune_model(best_model,
                      n_iter = 10,
                      optimize='recall', #the best model is selected based on the recall metric
                      search_library='optuna',
                      custom_grid = params,
                      #return_tuner=True,
                      choose_better=True)

# take the better ones. 
tuned_model = tuned_model_optuna

# confusion matrix of tuned model
plot_model(tuned_model , plot = 'confusion_matrix', plot_kwargs = {'percent' : True})


# show the list of the most important features for the model 
plot_model(tuned_model, plot = 'feature_all')


# performance in test data set

predict_model(tuned_model)

# test again with the saved test data set
test.drop('converted', axis = 1, inplace = True)

predict_model(tuned_model, data = test)





######### SAVE MODEL FOR AUTOMATION
save_model(tuned_model, 'tuned_model_file')


# for later use just load the model
saved_model = load_model('tuned_model_file')
# adjust threshold
predict_results = predict_model(saved_model, data = df_unseen_balanced,probability_threshold = 0.5)
