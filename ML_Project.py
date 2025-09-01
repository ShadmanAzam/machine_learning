import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns


df=pd.read_csv("/Users/syedshadmanazam/Student_Success.csv")

print(df.head())

print(df.describe())

print(df.info())

print(df.isnull)

print(df.dtypes)

print(f"row:{df.shape[0]} column: {df.shape[1]}")


le=LabelEncoder()

df["Internet"]=le.fit_transform(df["Internet"])
df["Passed"]=le.fit_transform(df["Passed"])

print(df.head())
print(df.dtypes)


features=["StudyHours","Attendance","PastScore","SleepHours"]
scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=scaler.fit_transform(df[features])

x=df_scaled[features]
y=df_scaled["Passed"]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


model=LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Classification Report")
print(classification_report(y_test,y_pred))

conf_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",
            xticklabels=["Fail","Pass"],yticklabels=["Fail","Pass"])


plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


print("--- Predict Your Result ---")

try:
    Study_hours=float(input("Enter Study Hours:"))
    attendence=float(input("Enter your  attendence:"))
    past_score=float(input("Enter your past score:"))
    sleep_hours=float(input("Enter your  sleep hours: "))
    
    user_input_df=pd.DataFrame([{
        'StudyHours':Study_hours,
        'Attendance':attendence,
        'PastScore':past_score,
        'SleepHours':  sleep_hours ,
    }])
    
    user_input_scaled=scaler.transform(user_input_df)
    
    pridiction=model.predict(user_input_scaled)[0]
    
    result= "Pass" if  pridiction ==1 else "Failed"
    print(f"Prediction Based on input : {result}")

except  Exception as e:
 print("n error occur")
