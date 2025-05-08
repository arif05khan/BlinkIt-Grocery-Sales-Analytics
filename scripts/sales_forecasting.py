from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BlinkIt Sales Forecasting") \
    .getOrCreate()

# Load the processed dataset
df = spark.read.parquet("data/processed/BlinkIt_Grocery_Dataset_Processed.parquet")

# Step 1: Aggregate sales by year
# Assuming there's a 'Sales' column (or 'Item Outlet Sales' in the dataset) and a 'Year' column
# If 'Year' isn't directly in the dataset, extract it from a date column if available
sales_by_year = df.groupBy(year(col("Date")).alias("Year")) \
                  .sum("Item Outlet Sales") \
                  .withColumnRenamed("sum(Item Outlet Sales)", "Total Sales") \
                  .orderBy("Year")

# Convert to Pandas for time series forecasting
sales_pd = sales_by_year.toPandas()
sales_pd.set_index("Year", inplace=True)

# Step 2: Prepare data for forecasting
# Ensure the data is sorted by year
sales_pd = sales_pd.sort_index()

# Step 3: Fit ARIMA model and forecast for 2023
# Using ARIMA(1,1,1) as a simple model; you can tune these parameters for better accuracy
model = ARIMA(sales_pd["Total Sales"], order=(1, 1, 1))
model_fit = model.fit()

# Forecast for the next year (2023)
forecast = model_fit.forecast(steps=1)
forecast_value = forecast.iloc[0]
forecast_year = 2023

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({"Year": [forecast_year], "Total Sales": [forecast_value]})
forecast_df.set_index("Year", inplace=True)

# Combine historical and forecast data
combined_df = pd.concat([sales_pd, forecast_df])

# Step 4: Save the forecast to Parquet
# Convert back to Spark DataFrame
forecast_spark_df = spark.createDataFrame(combined_df.reset_index())
forecast_spark_df.write.mode("overwrite").parquet("data/processed/Sales_Forecast.parquet")

# Step 5: Plot the historical sales and forecast
plt.figure(figsize=(10, 6))
plt.plot(sales_pd.index, sales_pd["Total Sales"], label="Historical Sales", marker="o")
plt.plot(combined_df.index[-1], combined_df["Total Sales"][-1], label="Forecast 2023", marker="x", color="red")
plt.title("BlinkIt Sales Forecast")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.legend()
plt.grid(True)
plt.savefig("docs/screenshots/forecast_results.png")
plt.close()

# Stop the Spark session
spark.stop()