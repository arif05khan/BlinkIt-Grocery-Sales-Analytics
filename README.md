BlinkIt Grocery Sales Analytics 🛒📊
Hey there! Welcome to my BlinkIt Grocery Sales Analytics project! I built this to dig into grocery sales data from BlinkIt, clean it up, scale it, and even forecast future sales for 2023. It’s a mix of data processing with Python and Spark, some forecasting magic with ARIMA, and cool visualizations in Power BI. I’m excited to share this with you—let’s dive in!
What’s This Project About? 🤔
I wanted to analyze sales data from BlinkIt to find trends and predict future sales. Here’s what I did:

Cleaned and processed the raw dataset using Spark (standardizing columns, filling missing values, etc.).
Scaled the data to 1 million rows to simulate a larger dataset.
Forecasted sales for 2023 using an ARIMA model in Python.
Visualized everything with a forecast plot and a Power BI dashboard.

<img width="630" alt="Dashboard" src="https://github.com/user-attachments/assets/22711613-c165-46b4-8c8a-2ccf996cf5cb" />


You’ll find all the code, results, and instructions here to run it yourself!
What’s in the Project Folder? 📂
Here’s a quick look at the project structure:

scripts/data_processing.py: The script that cleans and scales the dataset.
scripts/sales_forecasting.py: The script that forecasts sales and creates a plot.
data/raw/: Where you’ll place the raw dataset (you’ll need to download it—more on that below).
data/processed/: Where processed and forecast data gets saved (not in GitHub to save space).
data/synthetic/: Where the scaled dataset goes (also not in GitHub).
docs/screenshots/: Contains the sales forecast plot (forecast_results.png).
requirements.txt: Lists all the Python packages you’ll need.

What You’ll Need to Run This 🛠️
Before we get started, here’s what you’ll need on your machine:

Python 3.8 or higher: To run the scripts.
Apache Spark 3.5.0: For data processing.
Java 11: Spark needs this.
Power BI Desktop: To check out the dashboard.
Windows: I built this on Windows, but you can tweak paths for other operating systems.

Let’s Set It Up! 🚀
Follow these steps to get everything running. It might look like a lot, but I’ll walk you through it!

Clone the Project:First, grab the code from GitHub:
git clone https://github.com/arif05khan/BlinkIt-Grocery-Sales-Analytics.git
cd BlinkIt-Grocery-Sales-Analytics


Set Up a Virtual Environment:Let’s keep things tidy with a virtual environment:
python -m venv venv
venv\Scripts\activate  # On Windows (use `source venv/bin/activate` on Mac/Linux)


Install the Required Packages:Install everything listed in requirements.txt:
pip install -r requirements.txt


Set Up Apache Spark:

Download Spark 3.5.0 (spark-3.5.0-bin-hadoop3) from the Apache Spark website.
Unzip it to C:\spark\spark-3.5.0-bin-hadoop3 (or wherever you prefer).
Set the SPARK_HOME environment variable:
On Windows: Go to System Properties > Environment Variables > Add SPARK_HOME=C:\spark\spark-3.5.0-bin-hadoop3.




Set Up Java:

Install Java 11 (I used Adoptium’s version).
Set the JAVA_HOME environment variable:
On Windows: Add JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.0.21.9-hotspot (adjust based on your Java path).




Set Up Winutils (Windows Only):

Since I’m on Windows, Spark needs winutils.exe to work with Hadoop.
Download winutils.exe for Hadoop 3 (you can find it on GitHub or other sources—just make sure it’s legit).
Place it in C:\Users\Arif Rasul Khan\Downloads\hadoop\bin.
Set the HADOOP_HOME environment variable: HADOOP_HOME=C:\Users\Arif Rasul Khan\Downloads\hadoop.
Create a temporary directory for Hadoop:mkdir "C:\Users\Arif Rasul Khan\tmp\hive"
"C:\Users\Arif Rasul Khan\Downloads\hadoop\bin\winutils.exe" chmod -R 777 "C:\Users\Arif Rasul Khan\tmp\hive"


Set the HADOOP_TMP_DIR environment variable: HADOOP_TMP_DIR=C:\Users\Arif Rasul Khan\tmp\hive.


Get the Dataset and Power BI Dashboard:

I didn’t upload the dataset and dashboard to GitHub because they’re big files. Instead, you can download them from these links:
Dataset: BlinkIt_Grocery_Dataset.xlsx (replace with your Google Drive link).
Power BI Dashboard: BlinkIt_Sales_Dashboard.pbix (replace with your Google Drive link).


Place BlinkIt_Grocery_Dataset.xlsx in the data/raw/ folder.
Convert it to CSV (since the script expects a CSV file):
Open the .xlsx file in Excel.
Go to File > Save As, choose CSV (Comma delimited), and save it as data/raw/BlinkIt_Grocery_Dataset.csv.





How to Run the Project 🏃‍♂️
Now that everything’s set up, let’s run the scripts!

Process the Data:This script cleans the dataset, fills in missing values, and scales it to 1 million rows:
python scripts/data_processing.py


It’ll create:
data/processed/BlinkIt_Grocery_Dataset_Processed.parquet: The cleaned dataset.
data/synthetic/BlinkIt_Grocery_Dataset_Synthetic.parquet: The scaled dataset.




Forecast Sales:This script analyzes the cleaned data, forecasts sales for 2023, and makes a plot:
python scripts/sales_forecasting.py


It’ll create:
data/processed/Sales_Forecast.parquet: The forecast data.
docs/screenshots/forecast_results.png: A plot showing historical sales and the 2023 forecast.




Check Out the Power BI Dashboard:

Open BlinkIt_Sales_Dashboard.pbix in Power BI Desktop.
Import the Parquet files (data/processed/BlinkIt_Grocery_Dataset_Processed.parquet and data/processed/Sales_Forecast.parquet) to see the interactive visuals.



What I Found 📈
Here’s a quick look at the results:

Data Processing: I standardized the Item Fat Content column (e.g., changed "low fat" to "Low Fat") and filled in missing Item Weight values using the median for each Item Type. I also scaled the dataset to 1 million rows to simulate a larger dataset.
Sales Forecast: I used an ARIMA model to predict sales for 2023 based on historical data. Here’s the plot I got:
Power BI Dashboard: The dashboard lets you explore sales trends, item categories, and the forecast interactively. It’s pretty neat to play around with!

A Few Things to Keep in Mind ⚠️

The large files (Parquet, .xlsx, .pbix) aren’t in the GitHub repo to keep things lightweight. Make sure to download them using the links above.
I built this on Windows, so if you’re on Mac or Linux, you might need to adjust file paths (e.g., use / instead of \).
If you run into any issues with Spark, double-check your environment variables (SPARK_HOME, JAVA_HOME, HADOOP_HOME, etc.).

What’s Next? 🚀
I’m pretty happy with how this turned out, but there’s always room to improve! Maybe I’ll add more advanced forecasting models or dive deeper into sales by region. If you have any ideas or run into issues, feel free to open an issue on GitHub—I’d love to hear from you!

Thanks for checking out my project! 😊
