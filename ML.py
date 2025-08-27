import pandas as pd
import numpy as np

# # data = {
# #     'Name':['shadman','azam','ayan','usman'],
# #     'Age':[20,None,35,None],
# #     'Salary':[50,60,30,None]
    
# # }


# # df=pd.DataFrame(data)

# # # print(df)


# # # print(df.isnull().sum())



# # # df_drop=df.dropna()

# # # print(df_drop)
# # print(df.isnull().mean() * 100)

# # df['Age'].fillna(df['Age'].mean(), inplace=True)
# # df['Salary'].fillna(df['Salary'].mean(), inplace=True)


# # print(df)


# # from sklearn.preprocessing import LabelEncoder


# # df = pd.read_csv('/Users/syedshadmanazam/data.csv')

# # # print(df.head())

# # df_label=df.copy()

# # le = LabelEncoder()
# # lee = LabelEncoder()

# # print('/n Label Encoded Data')
# # df_label['Gender_Encoded']=le.fit_transform(df_label['Gender'])
# # df_label['Passed_Encoded']=lee.fit_transform(df_label['Passed'])

# # print(df_label[['Name','Gender','Gender_Encoded','Passed','Passed_Encoded']]) 




# # df_Encoded=pd.get_dummies(df_label,columns=['City'])

# # print("/n One-hot Encoded Data")

# # #Assignment convert bool city into binary
# # for col in df_Encoded.columns:
# #     if df_Encoded[col].dtype =='bool':
# #         df_Encoded[col]=df_Encoded[col].astype(int)
        
# # print(df_Encoded)


# from sklearn.preprocessing import StandardScaler,MinMaxScaler
# from sklearn.model_selection import train_test_split


# data={
#     'Studyhours':[1,2,3,4,5],
#     'TestScore':[40,50,60,70,80]
# }

# df=pd.DataFrame(data)

# Standard_scaler= StandardScaler()
# Standard_scaler=Standard_scaler.fit_transform(df)

# MinMax_Scaler=MinMaxScaler()
# MinMax_Scaler=MinMax_Scaler.fit_transform(df)

# # print(pd.DataFrame(Standard_scaler,columns=['StudyHours','TestScore']))
# print(pd.DataFrame(MinMax_Scaler,columns=['StudyHours','TestScore']))

# x =df[['Studyhours']]
# y =df[['TestScore']]

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# print("train data")

# print(x_train)
# print(y_train)

# print("test data")

# print(x_test)
# print(y_test)


# from sklearn.linear_model import LinearRegression

# model = LinearRegression()

# x=[[1],[2],[3],[4],[5],[6],[7]]
# y=[10,20,25,37,42,49,54]

# model.fit(x,y)

# Hours=float(input(" Enter number of study hours:"))

# prediction_marks = model.predict([[Hours]])

# print(f"Based on your { Hours} hours of study ,You may score around {prediction_marks}")

# from sklearn.linear_model import LogisticRegression

# model=LogisticRegression()

# x=[[1],[2],[3],[4],[5],[6]] #hours of study
# y=[0,0,1,1,1,1]

# model.fit(x,y)

# hours=float(input("Enter no of hours you studied:"))

# prediction = model.predict([[hours]])[0]

# if prediction ==1:
#     print(f"you are likely to pass, {prediction}")
# else:
#     print(f"you are likely to fail ,{prediction}")


# from sklearn.neighbors import KNeighborsClassifier

# x=[
#     [180,7],
#     [200,7.5],
#     [250,8],
#     [300,8.5],
#     [330,9],
#     [360,9.5]  
# ]

# y=[0,0,0,1,1,1]

# model = KNeighborsClassifier(n_neighbors=3)

# model.fit(x,y)

# weight=float(input("Enter the weight in grams:"))
# size=float(input("Enter the size in cm:"))

# prediction= model.predict([[weight,size]])[0]

# if prediction ==0:
#     print("This is likely an Apple")
# else:
#     print("This is likely an Orange")

# from sklearn.tree import DecisionTreeClassifier

# x=[
#     [7,2],#apple
#     [8,3],#apple
#     [9,8],#orange
#     [10,9]#orange
# ]

# y=[0,0,1,1]

# model=DecisionTreeClassifier()

# model.fit(x,y)

# size=float(input("Enter the fruit size in cm: "))
# shade=float(input("Enter the color shade: "))

# result = model.predict([[size,shade]])[0]

# if result==0:
#     print(f"this is an apple,{result}")
# else:
#     print(f"this is an orange,{result}")


# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# y_true=[1,0,1,1,0,1,0]

# y_pred=[1,0,1,0,0,1,1]

# print("Accuracy:", accuracy_score(y_true,y_pred))
# print("Precision:",precision_score(y_true,y_pred))
# print("Recall: ", recall_score(y_true,y_pred))
# print("F1_score:", f1_score(y_true,y_pred))


# from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
# import numpy as np

# real_score=[90,60,80,100]

# predicted_score=[85,70,70,95]

# mae=mean_absolute_error(real_score,predicted_score)
# mse=mean_squared_error(real_score,predicted_score)
# rmse=root_mean_squared_error(real_score,predicted_score)

# print("MSE:", mae)
# print("MAE:",mae)
# print("RMSE:",rmse)


# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error

# data=pd.read_csv("/Users/syedshadmanazam/srudent.csv")

# x=data[["Hours"]] #double brackets = 2D input
# y=data["Score"] #target Column


# model=LinearRegression()

# model.fit(x,y)

# predicted_score=model.predict(x)

# mae=mean_absolute_error(y,predicted_score)
# mse=mean_squared_error(y,predicted_score)
# rmse=root_mean_squared_error(y,predicted_score)

# print("MAE:",mae)
# print("MSE:",mse)
# print("RMSE:",rmse)

# new_prediction=float(input("Enetr a hours:"))
# new_pred=model.predict([[new_prediction]])
# print(f"prediction for {new_prediction}hours is {new_pred}")


