import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva", "Frank"],
    "Marks": [85.0, 92.0, np.nan, 78.0, 95.0, 88.0], 
    "Attendance": [90, 95, 80, 70, 98, 85],
}
df = pd.DataFrame(data)
print("--- Original Dataset with Missing Value ---")
print(df)
print("\n" + "=" * 40 + "\n")
mean_value=df["Marks"].mean()
df["Marks"]=df["Marks"].fillna(mean_value)
print("--- Modified Dataset with Missing Value ---")
print(df)
print("\n" + "=" * 40 + "\n")
mean_value=df["Marks"].mean()
median=df["Marks"].median()
sd=df["Marks"].std()
mi=df["Marks"].min()
ma=df["Marks"].max()
print("mean:",mean_value)
print("medain:",median)
print("standard deviation:",sd)
print("minimum:",mi)
print("maximum:",ma)
plt.bar(df["Name"], df["Marks"], color="skyblue")
plt.title("Student Marks")
plt.xlabel("Names")
plt.ylabel("Marks")
plt.show()
plt.hist(df["Marks"], bins=5, color="lightgreen", edgecolor="black")
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.show()
numeric_df = df[["Marks", "Attendance"]]
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
high=df[df["Marks"]==ma]["Name"].values[0]
print("the student with highest marks",high)
correlation=df["Attendance"].corr(df["Marks"])
print("correlation is:",correlation)