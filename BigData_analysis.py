# crime_analysis.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, year, month, when, concat_ws

# Start a Spark Session
spark = SparkSession.builder.appName("CrimeAnalysis").getOrCreate()

# Load Data
df = spark.read.csv("Crime_Data_from_2020_to_Present.csv", header=True, inferSchema=True)

# Clean and Format
df = df.withColumnRenamed("AREA NAME", "Area_Name") \
       .withColumnRenamed("Crm Cd Desc", "Crime_Description") \
       .withColumnRenamed("Vict Age", "Victim_Age") \
       .withColumnRenamed("Vict Sex", "Victim_Sex") \
       .withColumnRenamed("Vict Descent", "Victim_Descent") \
       .withColumn("YearMonth", concat_ws("-", year(col("DATE OCC")), month(col("DATE OCC"))))

# Trend Over Time
df_monthly_crime = df.groupBy("YearMonth").count().orderBy("YearMonth")
pdf_monthly_crime = df_monthly_crime.toPandas()
plt.figure(figsize=(12, 5))
plt.plot(pdf_monthly_crime["YearMonth"], pdf_monthly_crime["count"], marker='o')
plt.xticks(rotation=45)
plt.title("Crime Trends Over Time")
plt.show()

# Most Dangerous Areas
df_violent_area = df.groupBy("Area_Name").count().orderBy(col("count").desc())
pdf_violent_area = df_violent_area.toPandas()
plt.figure(figsize=(12, 5))
sns.barplot(x="Area_Name", y="count", data=pdf_violent_area.head(10))
plt.xticks(rotation=45)
plt.title("Top 10 Most Dangerous Areas")
plt.show()

# Time of Day Analysis
df_time = df.withColumn("Time_of_Day",
    when((col("TIME OCC") >= 0) & (col("TIME OCC") < 600), "Midnight - Morning")
    .when((col("TIME OCC") >= 600) & (col("TIME OCC") < 1200), "Morning - Noon")
    .when((col("TIME OCC") >= 1200) & (col("TIME OCC") < 1800), "Afternoon - Evening")
    .otherwise("Evening - Midnight"))

pdf_time = df_time.groupBy("Time_of_Day").count().toPandas()
plt.figure(figsize=(7, 7))
plt.pie(pdf_time["count"], labels=pdf_time["Time_of_Day"], autopct="%1.1f%%")
plt.title("Crime Distribution by Time of Day")
plt.show()

# Avg Victim Age by Crime Type
df_avg_age = df.groupBy("Crime_Description").agg({"Victim_Age": "avg"}) \
               .withColumnRenamed("avg(Victim_Age)", "Avg_Victim_Age") \
               .orderBy(col("Avg_Victim_Age").desc())
pdf_avg_age = df_avg_age.toPandas()
plt.figure(figsize=(12, 5))
sns.barplot(x="Avg_Victim_Age", y="Crime_Description", data=pdf_avg_age.head(10))
plt.title("Top 10 Crimes by Average Victim Age")
plt.show()

# Victim Age Group Distribution
df_age_group = df.withColumn("Age_Group",
    when(col("Victim_Age") < 18, "Under 18")
    .when((col("Victim_Age") >= 18) & (col("Victim_Age") <= 30), "18-30")
    .when((col("Victim_Age") >= 31) & (col("Victim_Age") <= 50), "31-50")
    .otherwise("Above 50"))

pdf_age = df_age_group.groupBy("Age_Group").count().toPandas()
plt.figure(figsize=(8, 5))
sns.barplot(x="Age_Group", y="count", data=pdf_age)
plt.title("Crime Count by Victim Age Group")
plt.show()

print("✅ Crime Analysis Completed Successfully!")
