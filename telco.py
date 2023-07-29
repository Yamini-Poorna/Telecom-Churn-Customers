##################################### All packages #####################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#taking data
telco=pd.read_excel("C:/Users/yamini/Desktop/GitHub/Hierarchical and Kmeans/Telco_customer_churn.xlsx")
telco               #7043 rows

telco.isna().sum()                   #no null values
telco.duplicated().sum()             #no duplicate values
telco.columns

#checking for zero variance varibales
telco.nunique(axis=0)               #count and quater is having same values. So, i will remove those columns
telco1=telco.drop(["Count","Quarter"],axis=1)
telco1.columns
telco1.nunique(axis=0)

# taking continuous values into a dataframe
#I am not taking customer id, as it is not needed
telco_con=telco1[['Number of Referrals',
       'Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download',
       'Monthly Charge', 'Total Charges', 'Total Refunds',
       'Total Extra Data Charges', 'Total Long Distance Charges',
       'Total Revenue']]

########################################## One hot encoding for categorical values
telco1.columns

enc=OneHotEncoder(handle_unknown="ignore")

enc_var=pd.DataFrame(enc.fit_transform(telco1[['Referred a Friend','Offer', 'Phone Service','Multiple Lines',
'Internet Service', 'Internet Type','Online Security', 'Online Backup', 'Device Protection Plan',
'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
'Payment Method']]).toarray())

enc_var.head(5)

#taking feature names 
enc.get_feature_names_out(['Referred a Friend','Offer', 'Phone Service','Multiple Lines',
'Internet Service', 'Internet Type','Online Security', 'Online Backup', 'Device Protection Plan',
'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
'Payment Method'])

#changing names for discrete variables
enc_var=enc_var.rename(columns={0:'Referred a Friend_No', 1:'Referred a Friend_Yes', 2:'Offer_None',
       3:'Offer_Offer A', 4:'Offer_Offer B', 5:'Offer_Offer C', 6:'Offer_Offer D',
       7:'Offer_Offer E', 8:'Phone Service_No', 9:'Phone Service_Yes',
       10:'Multiple Lines_No', 11:'Multiple Lines_Yes', 12:'Internet Service_No',
       13:'Internet Service_Yes', 14:'Internet Type_Cable', 15:'Internet Type_DSL',
       16:'Internet Type_Fiber Optic', 17:'Internet Type_None',
       18:'Online Security_No', 19:'Online Security_Yes', 20:'Online Backup_No',
       21:'Online Backup_Yes', 22:'Device Protection Plan_No',
       23:'Device Protection Plan_Yes', 24:'Premium Tech Support_No',
       25:'Premium Tech Support_Yes', 26:'Streaming TV_No', 27:'Streaming TV_Yes',
       28:'Streaming Movies_No', 29:'Streaming Movies_Yes',
       30:'Streaming Music_No', 31:'Streaming Music_Yes', 32:'Unlimited Data_No',
       33:'Unlimited Data_Yes', 34:'Contract_Month-to-Month',
       35:'Contract_One Year', 36:'Contract_Two Year', 37:'Paperless Billing_No',
       38:'Paperless Billing_Yes', 39:'Payment Method_Bank Withdrawal',
       40:'Payment Method_Credit Card', 41:'Payment Method_Mailed Check'
                                      })
enc_var             #this is discrete variables dataset

################################ joining both continuous and discrete data
telco2=telco_con.join(enc_var)
telco2.head(2)

telco2.isna().sum() #no null values
telco2.duplicated().sum()    #no duplicate values
################################ Oultlier treatment for continuous variables
telco2.columns

"""'Number of Referrals', 'Tenure in Months',
       'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download',
       'Monthly Charge', 'Total Charges', 'Total Refunds',
       'Total Extra Data Charges', 'Total Long Distance Charges',
       'Total Revenue'"""

telco3=telco2           #moving telco2 dataframe into telco3 

#Boxplot for number of referrals
plt.boxplot(telco3["Number of Referrals"])
plt.title("boxplot")
plt.show()

print("left side values: ", telco3["Number of Referrals"].mean()-3*telco3["Number of Referrals"].std())
print("right side values: ", telco3["Number of Referrals"].mean()+3*telco3["Number of Referrals"].std())
telco3[(telco3["Number of Referrals"]>10.955465001757492 ) | (telco3["Number of Referrals"]< -7.051730797583136)]

#winzorization for Number of Referrals
refferals_IQR=telco3["Number of Referrals"].quantile(0.75)-telco3["Number of Referrals"].quantile(0.25)
lower_limit_refferals=telco3["Number of Referrals"].quantile(0.25)-(1.5*refferals_IQR)
upper_limit_refferals=telco3["Number of Referrals"].quantile(0.75)+(1.5*refferals_IQR)

telco3["Number of Referrals"]=pd.DataFrame(np.where(telco3["Number of Referrals"]>upper_limit_refferals,upper_limit_refferals,np.where(telco3["Number of Referrals"]<lower_limit_refferals,lower_limit_refferals,telco3["Number of Referrals"])))

#Boxplot for Tenure in months
plt.boxplot(telco3["Tenure in Months"])
plt.title("Boxplot for tenure in months")
plt.show()
#no outliers

#Boxplot for 'Avg Monthly Long Distance Charges'
plt.boxplot(telco3['Avg Monthly Long Distance Charges'])
plt.title("boxplot")
plt.show()
#no outliers

#Boxplot for avg monthly gb download
plt.boxplot(telco3["Avg Monthly GB Download"])
plt.title("boxplot")
plt.show()

print("left side values: ", telco3["Avg Monthly GB Download"].mean()-3*telco3["Avg Monthly GB Download"].std())
print("right side values: ", telco3["Avg Monthly GB Download"].mean()+3*telco3["Avg Monthly GB Download"].std())
telco3[(telco3["Avg Monthly GB Download"]>81.77222654376759 ) | (telco3["Avg Monthly GB Download"]< -40.7414158097054)]

#Average monthly download GB is around 85 which is suspicious. So winsorizing all 91 rows
#Winsorization for avg monthly gb download
IQR_GB=telco3['Avg Monthly GB Download'].quantile(0.75)-telco3['Avg Monthly GB Download'].quantile(0.25)
lower_limit_GB=telco3['Avg Monthly GB Download'].quantile(0.25)-(IQR_GB*1.5)
upper_limit_GB=telco3['Avg Monthly GB Download'].quantile(0.75)+(IQR_GB*1.5)

telco3['Avg Monthly GB Download']=pd.DataFrame(np.where(telco3['Avg Monthly GB Download']>upper_limit_GB,upper_limit_GB,np.where(telco3['Avg Monthly GB Download']<lower_limit_GB,lower_limit_GB,telco3['Avg Monthly GB Download'])))

#Boxplot for Monthly Charge
plt.boxplot(telco3["Monthly Charge"])
plt.title("boxplot")
plt.show()
#no outliers

#Boxplot for Total Charges
plt.boxplot(telco3["Total Charges"])
plt.title("boxplot")
plt.show()
#no outliers

#Boxplot for Total Refunds
plt.boxplot(telco3["Total Refunds"])
plt.title("boxplot")
plt.show()
#if we are doing winsorization for total refunds it becomes all values 0. Because only 266 people out out 7041 got refund. So i want to keep as it is.

#Boxplot for Total Extra Data Charges
plt.boxplot(telco3["Total Extra Data Charges"])
plt.title("boxplot")
plt.show()
#259 rows are different out of nearly 7000 values. So no need to do outlier treatment

#Boxplot for Total Long Distance Charges
plt.boxplot(telco3["Total Long Distance Charges"])
plt.title("boxplot")
plt.show()

print("left side values: ", telco3["Total Long Distance Charges"].mean()-3*telco3["Total Long Distance Charges"].std())
print("right side values: ", telco3["Total Long Distance Charges"].mean()+3*telco3["Total Long Distance Charges"].std())
telco3[(telco3["Total Long Distance Charges"]>3289.0794260744005 ) | (telco3["Total Long Distance Charges"]< -1790.8809027178768)]

#winzorization for Total Long Distance Charges
refferals_IQR=telco3["Total Long Distance Charges"].quantile(0.75)-telco3["Total Long Distance Charges"].quantile(0.25)
lower_limit_refferals=telco3["Total Long Distance Charges"].quantile(0.25)-(1.5*refferals_IQR)
upper_limit_refferals=telco3["Total Long Distance Charges"].quantile(0.75)+(1.5*refferals_IQR)

telco3["Total Long Distance Charges"]=pd.DataFrame(np.where(telco3["Total Long Distance Charges"]>upper_limit_refferals,upper_limit_refferals,np.where(telco3["Total Long Distance Charges"]<lower_limit_refferals,lower_limit_refferals,telco3["Total Long Distance Charges"])))

#boxplot for total revenue
plt.boxplot(telco3["Total Revenue"])
plt.title("boxplot")
plt.show()

print("left side values: ", telco3["Total Revenue"].mean()-3*telco3["Total Revenue"].std())
print("right side values: ", telco3["Total Revenue"].mean()+3*telco3["Total Revenue"].std())
telco3[(telco3["Total Revenue"]>11629.992680334672 ) | (telco3["Total Revenue"]< -5561.234568734493)]
#telco1.iloc[5491]                       #just checking the values

#winsorization for total revenue
revenue_IQR=telco3["Total Revenue"].quantile(0.75)-telco3["Total Revenue"].quantile(0.25)
lower_limit_revenue=telco3["Total Revenue"].quantile(0.25)-(1.5*revenue_IQR)
upper_limit_revenue=telco3["Total Revenue"].quantile(0.75)+(1.5*revenue_IQR)

telco3["Total Revenue"]=pd.DataFrame(np.where(telco3["Total Revenue"]>upper_limit_revenue,upper_limit_revenue,np.where(telco3["Total Revenue"]<lower_limit_revenue,lower_limit_revenue,telco3["Total Revenue"])))

################################  Normalization for entire dataset
telco4_norm=pd.DataFrame(normalize(telco3))
telco5_norm=pd.DataFrame(telco4_norm)
telco5_norm              #52 columns

####################################### Doing PCA
#Choosing the right number of dimensions. So first applying the PCA to original number of dimensions.
pca_52=PCA(n_components=52,random_state=35)
pca_52.fit(telco5_norm)
pca_52_telco5_norm=pca_52.transform(telco5_norm)

#variance explained by 13 components
pca_52.explained_variance_ratio_*100

#checking the sum of variance of all components should be 100
sum(pca_52.explained_variance_ratio_*100)

#checking the cumulative sum
np.cumsum(pca_52.explained_variance_ratio_*100)

#I want to take first 10 components as it gives 99.91%
#creating a elbow plot for pca components cumulative sum
plt.plot(np.cumsum(pca_52.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("explained variance")
plt.show()              #elbow curve is also showing to take 9 components

#So lets take 9 pca components and build the pca again
pca_9=PCA(n_components=9,random_state=35)
pca_9.fit(telco5_norm)
pca_9_telco5_norm=pca_9.transform(telco5_norm)

#Create data frame using 9 principal components
telco6_pca=pd.DataFrame(pca_9_telco5_norm,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9'])
telco6_pca.head(5)
telco6_pca.shape                       #7043 rows and 9 columns

####################################### K-means clustering #####################################
#taking kmeans parameters for elbow curve
kmeans_kwargs={"init":"k-means++",
               "max_iter":300,
               "n_init":10,                        #default=10
               "random_state":49
}

sse=[]

for k in range(1,11):
    kmeans=KMeans(n_clusters=k,**kmeans_kwargs)
    kmeans.fit(telco6_pca)
    sse.append(kmeans.inertia_)
     
#elbow curve
plt.plot(range(1,11),sse)
plt.xticks(range(1,11))
plt.xlabel("number of clusters")
plt.ylabel("sse")
plt.show()

#finding the elbow
k1=KneeLocator(range(1,11),sse,curve="convex",direction="decreasing")
k1.elbow                #4 clusters we have to take according to the elbow curve

#doing kmeans
kmeans=KMeans(init="k-means++", n_init=1, max_iter=300, n_clusters=4, random_state=49)
kmeans.fit(telco6_pca)
kmeans.inertia_                   #160.79
kmeans.n_iter_                     #10

#kmeans clusters
kmeans.labels_

kmeans_silhouette = silhouette_score(telco6_pca, kmeans.labels_).round(2)
kmeans_silhouette             #0.43
#silhouette score is good 

######################################### Hierarchical clustering ########################
#finding distance
z=linkage(telco6_pca,metric="euclidean", method="complete")

#dendrogram
plt.figure(figsize=(15,8)) 
plt.title("hierarchical clustering dendrogram")
plt.xlabel("index")
plt.ylabel("distance")
dendrogram(z)
plt.show()

#According to dendrogram, we have to take 4 clusters.
telco_hier=AgglomerativeClustering(n_clusters=4,linkage="complete",affinity="euclidean").fit(telco6_pca)

#hierarchichal clustering labels
telco_hier.labels_ 

hier_silhouette = silhouette_score(telco6_pca, telco_hier.labels_).round(2)
hier_silhouette                 #0.36
#hierarichal clustering silhouette score is less than kmeans. 

########################################### Kmeans clusters
#kmeans having silhouette score much better than agglomerative clustering. So i am taking kmeans
#taking the kmeans labels
kmeans.labels_

#making labels into a seperate column
clusters=pd.Series(kmeans.labels_)
clusters

#adding clusters varaible to dataframe
telco1["clusters"]=clusters
telco1.head(2)

#taking overall cluster results using mean with groupby clusters
telco_clusters=telco1.iloc[:,0:28].groupby(telco1.clusters).mean().round(4)
telco_clusters

#saving the dataset with clusters
telco1.to_csv("telco_cluster_churn_labels.csv")











