from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, median

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BlinkIt Data Processing") \
    .getOrCreate()

# Load the dataset
df = spark.read.option("header", "true").option("inferSchema", "true").csv("data/raw/BlinkIt_Grocery_Dataset.csv")

# Step 1: Standardize 'Item Fat Content' column
df = df.withColumn("Item Fat Content", 
                   when(col("Item Fat Content").isin(["LF", "low fat"]), "Low Fat")
                   .when(col("Item Fat Content").isin(["reg"]), "Regular")
                   .otherwise(col("Item Fat Content")))

# Step 2: Fill missing 'Item Weight' values with median per 'Item Type'
# Calculate median 'Item Weight' per 'Item Type'
median_weights = df.groupBy("Item Type").agg(median("Item Weight").alias("median_weight"))

# Join the median weights back to the original dataframe and fill missing values
df = df.join(median_weights, on="Item Type", how="left") \
       .withColumn("Item Weight", 
                   when(col("Item Weight").isNull(), col("median_weight"))
                   .otherwise(col("Item Weight"))) \
       .drop("median_weight")

# Step 3: Save the cleaned dataset
df.write.mode("overwrite").parquet("data/processed/BlinkIt_Grocery_Dataset_Processed.parquet")

# Step 4: Scale the dataset to 1 million rows (synthetic data generation)
# Calculate the scaling factor to reach 1 million rows
current_rows = df.count()
target_rows = 1_000_000
scaling_factor = target_rows // current_rows

# Replicate the dataset to approximate 1 million rows
synthetic_df = df
for _ in range(scaling_factor - 1):
    synthetic_df = synthetic_df.union(df)

# If we overshoot or undershoot, trim or add rows to get closer to 1 million
final_rows = synthetic_df.count()
if final_rows > target_rows:
    # Randomly sample to reduce to 1 million
    fraction = target_rows / final_rows
    synthetic_df = synthetic_df.sample(fraction=fraction, seed=42)
elif final_rows < target_rows:
    # Add more rows to reach 1 million
    remaining_rows = target_rows - final_rows
    fraction = remaining_rows / current_rows
    additional_rows = df.sample(fraction=fraction, seed=42)
    synthetic_df = synthetic_df.union(additional_rows)

# Save the synthetic dataset
synthetic_df.write.mode("overwrite").parquet("data/synthetic/BlinkIt_Grocery_Dataset_Synthetic.parquet")

# Stop the Spark session
spark.stop()