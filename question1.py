#%%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

#read dataset
data = pd.read_csv('data/dataClustering.csv')

#calculate optimal number of clusters based on silhouette scores for Kmeans clustering (assume at least 2 clusters)
silhouette_scores_df = pd.DataFrame(columns=['Clusters','Silhouette_scores'])
for k in range(2,12):
    kmeans = KMeans(n_clusters=k,init='k-means++',random_state=0).fit(data)
    labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, labels)
    silhouette_scores_df = silhouette_scores_df.append({'Clusters':k,'Silhouette_scores':silhouette_avg},ignore_index=True)

#find index of the maximum silhouette score
index_of_max_score = silhouette_scores_df['Silhouette_scores'].idxmax()
optimal_clusters = int(silhouette_scores_df['Clusters'][index_of_max_score])
print('''The optimal number of clusters based on Silhouette score is equal to '''+str(optimal_clusters))

#reset labels based on the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters,init='k-means++',random_state=0).fit(data)
labels = kmeans.fit_predict(data)
#assign labels to data points
data['Cluster'] = labels

#split data into train and test sets randomly based on 70/30 train/test ratio
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1],
                                                    shuffle = True, 
                                                    test_size=0.3, 
                                                    random_state=1)

#calculate baseline multinomial model
baseline_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
baseline_model.fit(X_train,y_train)

#calculate predictions
y_predicted = baseline_model.predict(X_test)

#use decision tree ML model
dtc = tree.DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
y_predicted_tree = dtc.predict(X_test)

#create a dataframe to find if the predicted cluster matched the assigned cluster for both models
evaluate_df = pd.DataFrame([y_predicted - y_test, y_predicted_tree - y_test],index=['Multilogit','Tree']).T

#any misassigned cluster will be a value other than 0
wrong_points_multilogit = len(evaluate_df[evaluate_df['Multilogit']!=0])
wrong_points_tree = len(evaluate_df[evaluate_df['Tree']!=0])

if wrong_points_multilogit > wrong_points_tree:
    print('The Decision Tree model is better because it has '+str(wrong_points_tree)+''' wrong test points while the Multilogit has '''+str(wrong_points_multilogit)+''' wrong test points.''')
else:
    print('The Multilogit model is better because it has '+str(wrong_points_multilogit)+''' wrong test points while the Decision tree model has '''+str(wrong_points_tree)+''' wrong test points.''')

# %%
