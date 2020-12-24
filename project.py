#imports libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

#reads csv dataset and drops any unnecessary parameters
df = pd.read_csv('anime_cleaned.csv',encoding='utf-8_sig')
df = df.drop(['title_english','title_japanese','title_synonyms','image_url','status',
        'aired_string','aired','rank','popularity','background','premiered','related',
        'opening_theme','ending_theme','title','producer','licensor','duration','airing','broadcast'],axis=1)
#at this point, the parameters that are left are: anime_id, type, source, episodes, rating, score, scored_by, members,
        #favorites, studio, genre, duration_min, aired_from_year

#fills null values with empty string
df.fillna("",inplace=True)

#converts strings into numbers so dataset can be processed
le = LabelEncoder()
for i in df.columns.values.tolist():
    df[i] = le.fit_transform(df[i])

#the 'score' is set as the target parameter, with x's parameters everything besides that
y = df['score']
x = df.drop(['score'],axis= 1)

#here, the processed data is temporarily saved in a new csv to check changes and/or errors
y_csv = y
y_csv = pd.concat([y_csv,x],axis=1)
y_csv.to_csv('temp.csv',encoding='utf-8')

#the dataset is split into training and testing sets, at a ratio of 8:2
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=54,shuffle=True)

#RandomForestClassifier is used to analyze data and return accuracy of model
clsf = RandomForestClassifier(n_estimators=100,min_samples_split=2,max_depth=2,random_state=8)
clsf.fit(x_train,y_train)
print('Score: ' + str(clsf.score(x_test,y_test)))

#Features influencing the target parameter are reordered by importance and displayed
feat_imp = clsf.feature_importances_
dic = dict(zip(x.columns,clsf.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))
