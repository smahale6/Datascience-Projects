import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px

# read the csv file into a pandas data frame
food_consumption  = pd.read_csv("food-consumption.csv")
# remove the 3 countries as mentioned in the homework instructions
food_consumption = food_consumption.loc[-food_consumption["Country"].isin(['Sweden', "Finland", "Spain"]),:]
# create a new dataframe with the columns required
lst_food_consumption = food_consumption.columns[1::]
# Store the country info in a data frame
Country = food_consumption["Country"]
#define a scaler to scale the data and get the mean
sc = StandardScaler() 
# Scale the data
food_consumption = sc.fit_transform(food_consumption.iloc[:,1:]- np.mean(food_consumption.iloc[:,1:], axis=0))
# create a covariance matrix with the data
cov_food_consumption = np.cov(food_consumption,rowvar=False)
#derive the eigen value and eigen vector
eig_value, eig_vector = np.linalg.eig(cov_food_consumption)
eig_value = eig_value[np.argsort(-eig_value)]
eig_vector = eig_vector[:,np.argsort(-eig_value)]
# calculate PCAs
PrincipalComponent_1 = food_consumption.dot(eig_vector[:,0].transpose()).real
PrincipalComponent_2 = food_consumption.dot(eig_vector[:,1].transpose()).real
# Assign df_PCA with the Country Labels
df_PCA = pd.DataFrame(data={"Country":Country, "PC_1":PrincipalComponent_1, "PC_2":PrincipalComponent_1}).reset_index().iloc[:,1:4]
# create a scatter plot of the 2 principal components and run kemans to get pattern
plot_1  = px.scatter(df_PCA, x='PC_1',y='PC_2',text="Country", color=KMeans(n_clusters=4).fit(df_PCA.iloc[:,1:3]).labels_.astype(object),color_continuous_scale='Bluered_r', title="Country - KMeans clusters",width=1200, height=600)
plot_1.update_traces(textposition='bottom center')
plot_1.update_layout(showlegend=False)
plot_1.show()
# get the 2 eigen vectors
PCA_v2 = eig_vector[:,0:2].real
# scatter plof of 2 dimensional reduced representation by country
plot_2 = sns.regplot(x='PC_1', y='PC_2', data=df_PCA, fit_reg=False)
# total numner of food items
count_food_use=20
# plot by country
for line in range(0,df_PCA.shape[0]):
    plot_2.text(df_PCA.PC_1[line], df_PCA.PC_2[line], df_PCA.Country[line], horizontalalignment='left', size='large', color='teal')
# sub plot the food items
for i in range(count_food_use):
    plt.arrow(0, 0, PCA_v2[i,0]*8.7, PCA_v2[i,1]*8.7,color = 'gray',linestyle='--', alpha = 0.25)
    plot_2.text(PCA_v2[i,0]*8.7, PCA_v2[i,1]*8.7, lst_food_consumption[i], color = 'green', size='small', ha = 'center', va = 'center')
plt.title('Scatter Plot - Reduced by Country')
plt.show()



