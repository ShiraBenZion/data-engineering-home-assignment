import math
import tempfile
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from pyspark.errors import AnalysisException
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.window import Window
import logging
import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
STACK_NAME = os.getenv("STACK_NAME")

logging.basicConfig(
    filename='stock_analysis.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger()


class DataValidator:
    @staticmethod
    def general_validation(df):
        try:
            if df.isEmpty():
                logger.error("Dataframe is empty")
                raise

            total_count = df.count()
            null_count = df.filter(col("ticker").isNull()).count()
            null_percentage = (null_count / total_count) * 100
            if null_percentage > 5:
                logger.error("Too many null values in the 'ticker' column")
                raise
            if 0 < null_percentage < 5:
                logger.warning("There are null values in 'ticker', which may affect analysis results")
                df = df.dropna(subset=["ticker"])

            logger.info("General validation succeeded")
            return df
        except Exception as e:
            logger.error(f"General validation failed: {e}")
            raise

    @staticmethod
    def correct_negative_volume(df):
        try:
            logger.info("Checking and correcting negative volume values")
            df = df.withColumn("volume", F.abs(F.col("volume")))
            return df
        except Exception as e:
            logger.error(f"Error correcting negative volume: {e}")
            raise

    @staticmethod
    def drop_nulls(df, columns):
        try:
            df = df.dropna(subset=columns)
            logger.info(f"Dropped rows with null values in columns: {columns}")
            return df
        except Exception as e:
            logger.error(f"Error while dropping nulls in columns {columns}: {e}")
            raise

    @staticmethod
    def drop_duplicates(df, columns):
        try:
            # Drop duplicate rows based on the specified columns
            df = df.dropDuplicates(columns)
            logger.info(f"Dropped duplicate rows based on columns: {columns}")
            return df
        except Exception as e:
            logger.error(f"Error while dropping duplicates in columns {columns}: {e}")
            raise


class SparkSessionCreationError(Exception):
    pass


class ErrorHandler:
    @staticmethod
    def create_error_df(error_message, spark):
        try:
            # Create dataframe with error's value
            error_data = [("ERROR", error_message)]
            error_columns = ["status", "error_message"]

            error_df = spark.createDataFrame(error_data, error_columns)
            return error_df
        except Exception as e:
            logger.error("Cant create an error result Dataframe")
            raise


class StockDataAnalysis:
    def __init__(self, input_path, output_bucket_path):
        try:
            logger.info("Starting Stock Data Analysis")
            try:
                self.spark = SparkSession.builder \
                    .appName("StockDataAnalysis").getOrCreate()

            except SparkSessionCreationError as e:
                logger.error(f"Spark session creation failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

            try:
                self.df = self.spark.read.csv(input_path, header=True, inferSchema=True)
                self.output_bucket_path = output_bucket_path

            except AnalysisException as e:
                logger.error(f"Error reading CSV file: {e}")
                raise ValueError(f"Failed to read CSV file at {input_path}") from e

            self.validator = DataValidator
            self.df = self.validator.general_validation(self.df)
            self.df = self.validator.drop_duplicates(self.df, ["Date", "volume"])

            # Calculate daily returns and store for use in multiple methods
            self.daily_returns_df = self.calculate_daily_returns()

            logger.info("Stock Data Analysis initialized successfully")

        except Exception as e:
            logger.error(f"Error Starting Stock Data Analysis: {e}")
            raise

    def calculate_daily_returns(self):
        try:
            logger.info("Calculating daily returns")

            # Clean nulls
            filtered_df = self.validator.drop_nulls(self.df, ["close"])

            # Calculate
            window_spec = Window.partitionBy("ticker").orderBy("Date")
            return_df = filtered_df.withColumn("Date", F.date_format("Date", "yyyy-MM-dd")) \
                .withColumn("previous_close", F.lag(F.col("close")).over(window_spec)) \
                .withColumn("daily_return", (F.col("close") - F.col("previous_close")) / F.col("previous_close"))

            logger.info("Daily returns calculated successfully")
            return return_df

        except Exception as e:
            logger.error(f"Error calculating daily returns: {e}")
            raise

    def calculate_daily_average(self):
        try:
            logger.info("Calculating daily average return")
            daily_returns_df = self.daily_returns_df

            # Filtering the columns for efficiency
            filtered_df = daily_returns_df.select("Date", "daily_return")

            # Clean nulls
            filtered_df = self.validator.drop_nulls(filtered_df, ["daily_return"])

            # Calculate
            average_daily_return = filtered_df.groupBy("Date") \
                .agg(F.mean("daily_return").alias("average_return")) \
                .orderBy("Date")
            logger.info("Daily average return calculated successfully")
            return average_daily_return

        except Exception as e:
            logger.error(f"Error calculating daily average return: {e}")
            return ErrorHandler.create_error_df(e, self.spark)

    def calculate_highest_average_worth(self):
        try:
            logger.info("Calculating highest average worth")

            # Clean nulls
            self.df = self.validator.drop_nulls(self.df, ["close", "volume"])

            # Correct negative volumes
            corrected_volume_df = self.validator.correct_negative_volume(self.df)

            # Filter the columns for efficiency
            corrected_volume_df = corrected_volume_df.select("close", "ticker", "volume")

            # Calculate
            worth_df = corrected_volume_df.withColumn("worth", F.col("close") * F.col("volume"))
            avg_worth = worth_df.groupBy("ticker") \
                .agg(F.mean("worth").alias("average_worth"))

            window_spec = Window.orderBy(F.col("average_worth").desc())
            ranked_df = avg_worth.withColumn("rank", F.row_number().over(window_spec))
            highest_worth_stock = ranked_df.filter(F.col("rank") == 1).drop("rank")
            logger.info("Highest average worth calculated successfully")

            return highest_worth_stock

        except Exception as e:
            logger.error(f"Error calculating highest average worth: {e}")
            return ErrorHandler.create_error_df(e, self.spark)

    def calculate_volatility(self):
        try:
            logger.info("Calculating volatility")
            daily_returns_df = self.daily_returns_df

            # Filter the columns for efficiency
            filtered_df = daily_returns_df.select("ticker", "volume", "daily_return")

            # Clean nulls
            filtered_df = self.validator.drop_nulls(filtered_df, ["daily_return"])

            # Calculate
            std_dev_df = filtered_df.groupBy("ticker") \
                .agg(F.stddev("daily_return").alias("standard_deviation"))

            std_dev_df = std_dev_df.withColumn("annualized_std_dev", F.col("standard_deviation") * math.sqrt(252))
            result_df = std_dev_df.select("ticker", "annualized_std_dev") \
                .withColumnRenamed("annualized_std_dev", "standard_deviation")

            logger.info("Volatility calculated successfully")
            return result_df.orderBy(F.col("standard_deviation").desc())

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return ErrorHandler.create_error_df(e, self.spark)

    def calculate_top_three_30_day_returns(self):
        try:
            logger.info("Calculating top three 30-day returns")

            # Filter the columns for efficiency
            filtered_df = self.df.select("ticker", "Date", "close")

            # Clean nulls
            filtered_df = self.validator.drop_nulls(filtered_df, ["close"])

            window_spec = Window.partitionBy("ticker").orderBy("Date").rowsBetween(-30, -1)
            filtered_df = filtered_df.withColumn("previous_30_close", F.avg("close").over(window_spec))

            filtered_df = self.validator.drop_nulls(filtered_df, ["previous_30_close"])

            # Calculate
            filtered_df = filtered_df.withColumn("percentage_increase",
                                                 (F.col("close") - F.col("previous_30_close")) / F.col(
                                                     "previous_30_close") * 100)
            top_returns_df = filtered_df.orderBy(F.col("percentage_increase").desc()).groupBy("ticker") \
                .agg(F.first("Date").alias("date"), F.first("percentage_increase").alias("percentage_increase")) \
                .orderBy(F.col("percentage_increase").desc()).limit(3)

            logger.info("Top three 30-day returns calculated successfully")
            return top_returns_df

        except Exception as e:
            logger.error(f"Error calculating top three 30-day returns: {e}")
            return ErrorHandler.create_error_df(e, self.spark)

    @staticmethod
    def save_results(dataframe, s3_path):
        try:
            logger.info(f"Saving results to {s3_path}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                local_file_path = tmp_file.name
                dataframe.toPandas().to_csv(local_file_path, index=False)

            s3 = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )

            s3_bucket = s3_path.split('/')[2]
            s3_key = '/'.join(s3_path.split('/')[3:])

            try:
                s3.upload_file(local_file_path, s3_bucket, s3_key)

            except NoCredentialsError as e:
                logger.error(f"Credentials are not provided: {e}")
                raise

            except PartialCredentialsError as e:
                logger.error(f"Partial credentials provided: {e}")
                raise

            except Exception as e:
                logger.error(f"Failed to upload to S3: {e}")
                raise

            logger.info(f"Results saved successfully to {s3_path}")

        except Exception as e:
            logger.error(f"Error saving results to {s3_path}: {e}")
            raise

    def run_analysis(self):
        try:
            logger.info("Running stock analysis")

            daily_average_df = self.calculate_daily_average()
            daily_return_path = f"{self.output_bucket_path}/daily_average.csv"
            self.save_results(daily_average_df, daily_return_path)

            highest_average_worth_df = self.calculate_highest_average_worth()
            highest_average_worth_path = f"{self.output_bucket_path}/highest_average_worth.csv"
            self.save_results(highest_average_worth_df, highest_average_worth_path)

            volatility_df = self.calculate_volatility()
            volatility_path = f"{self.output_bucket_path}/volatility.csv"
            self.save_results(volatility_df, volatility_path)

            top_three_30_days_return_df = self.calculate_top_three_30_day_returns()
            top_three_path = f"{self.output_bucket_path}/top_three_30_day_returns.csv"
            self.save_results(top_three_30_days_return_df, top_three_path)

            logger.info("Stock analysis completed successfully")

        except Exception as e:
            logger.error(f"Error running stock analysis: {e}")
            raise

        finally:
            # Ensuring that Spark session is stopped after the analysis
            if self.spark:
                logger.info("Stopping Spark session")
                self.spark.stop()


if __name__ == '__main__':
    run = StockDataAnalysis(r"./stocks_data.csv", f"s3a://{STACK_NAME}/analyzed_output")
    run.run_analysis()
