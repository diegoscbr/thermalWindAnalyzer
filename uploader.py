#!/usr/bin/env python3
"""
CSV to BigQuery Uploader
Uploads multiple CSV files to Google BigQuery with automatic schema detection
"""

import os
import glob
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import json
from datetime import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVToBigQueryUploader:
    def __init__(self, project_id, dataset_id, credentials_path=None):
        """
        Initialize the uploader
        
        Args:
            project_id (str): Your Google Cloud Project ID
            dataset_id (str): BigQuery dataset name
            credentials_path (str): Path to service account JSON file (optional if using environment variable)
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        
        # Initialize BigQuery client
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        try:
            self.client = bigquery.Client(project=project_id)
            logger.info(f"✅ Connected to BigQuery project: {project_id}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to BigQuery: {e}")
            raise
        
        # Create dataset if it doesn't exist
        self.create_dataset_if_not_exists()
    
    def create_dataset_if_not_exists(self):
        """Create the BigQuery dataset if it doesn't already exist"""
        dataset_ref = self.client.dataset(self.dataset_id)
        
        try:
            self.client.get_dataset(dataset_ref)
            logger.info(f"✅ Dataset '{self.dataset_id}' already exists")
        except NotFound:
            # Create the dataset
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"  # or your preferred location
            dataset.description = f"Dataset created by CSV uploader on {datetime.now()}"
            
            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"✅ Created dataset '{self.dataset_id}'")
    
    def clean_column_names(self, df):
        """
        Clean column names to be BigQuery compatible
        - Replace spaces and special characters with underscores
        - Convert to lowercase
        - Remove leading/trailing whitespace
        """
        df.columns = df.columns.str.strip()  # Remove whitespace
        df.columns = df.columns.str.lower()  # Convert to lowercase
        df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)  # Replace special chars
        df.columns = df.columns.str.replace(r'_+', '_', regex=True)  # Replace multiple underscores
        df.columns = df.columns.str.strip('_')  # Remove leading/trailing underscores
        
        return df
    
    def infer_bigquery_schema(self, df):
        """
        Infer BigQuery schema from pandas DataFrame
        """
        type_mapping = {
            'int64': 'INTEGER',
            'float64': 'FLOAT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'datetime64[ns, UTC]': 'TIMESTAMP',
            'object': 'STRING'  # Default for strings and mixed types
        }
        
        schema = []
        for column, dtype in df.dtypes.items():
            bq_type = type_mapping.get(str(dtype), 'STRING')
            
            # Handle datetime columns specifically
            if 'datetime' in str(dtype):
                bq_type = 'TIMESTAMP'
            
            schema.append(bigquery.SchemaField(column, bq_type))
        
        return schema
    
    def preprocess_dataframe(self, df, filename):
        """
        Preprocess DataFrame before uploading
        """
        logger.info(f"📊 Preprocessing {filename}: {len(df)} rows, {len(df.columns)} columns")
        
        # Clean column names
        df = self.clean_column_names(df)
        
        # Handle datetime columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert datetime-like strings
                try:
                    if df[col].str.contains(r'\d{4}-\d{2}-\d{2}', na=False).any():
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.info(f"  ⏰ Converted '{col}' to datetime")
                except:
                    pass
        
        # Handle missing values
        df = df.replace({float('inf'): None, float('-inf'): None})
        
        logger.info(f"✅ Preprocessed DataFrame: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def upload_csv_to_bigquery(self, csv_path, table_name=None, if_exists='replace'):
        """
        Upload a single CSV file to BigQuery
        
        Args:
            csv_path (str): Path to CSV file
            table_name (str): BigQuery table name (if None, uses filename without extension)
            if_exists (str): 'replace', 'append', or 'fail'
        """
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            logger.error(f"❌ File not found: {csv_path}")
            return False
        
        # Generate table name from filename if not provided
        if table_name is None:
            table_name = csv_path.stem.lower()
            # Clean table name
            table_name = table_name.replace('-', '_').replace(' ', '_')
        
        logger.info(f"📤 Uploading {csv_path.name} to table '{table_name}'...")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            if df.empty:
                logger.warning(f"⚠️  CSV file {csv_path.name} is empty, skipping...")
                return False
            
            # Preprocess
            df = self.preprocess_dataframe(df, csv_path.name)
            
            # Create table reference
            table_ref = self.client.dataset(self.dataset_id).table(table_name)
            
            # Configure job
            job_config = bigquery.LoadJobConfig()
            
            if if_exists == 'replace':
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            elif if_exists == 'append':
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            else:
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
            
            # Auto-detect schema or use inferred schema
            job_config.autodetect = True
            
            # Upload
            job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            job.result()  # Wait for completion
            
            # Get final table info
            table = self.client.get_table(table_ref)
            
            logger.info(f"✅ Successfully uploaded to '{table_name}': {table.num_rows} rows")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload {csv_path.name}: {e}")
            return False
    
    def upload_multiple_csvs(self, csv_directory, pattern="*.csv", if_exists='replace'):
        """
        Upload multiple CSV files from a directory
        
        Args:
            csv_directory (str): Directory containing CSV files
            pattern (str): File pattern to match (e.g., "*.csv", "weather_*.csv")
            if_exists (str): 'replace', 'append', or 'fail'
        """
        csv_path = Path(csv_directory)
        
        if not csv_path.exists():
            logger.error(f"❌ Directory not found: {csv_directory}")
            return
        
        # Find CSV files
        csv_files = list(csv_path.glob(pattern))
        
        if not csv_files:
            logger.warning(f"⚠️  No CSV files found matching pattern '{pattern}' in {csv_directory}")
            return
        
        logger.info(f"📁 Found {len(csv_files)} CSV files to upload...")
        
        # Track results
        successful_uploads = 0
        failed_uploads = 0
        
        for csv_file in csv_files:
            logger.info(f"\n{'='*50}")
            success = self.upload_csv_to_bigquery(csv_file, if_exists=if_exists)
            
            if success:
                successful_uploads += 1
            else:
                failed_uploads += 1
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 UPLOAD SUMMARY:")
        logger.info(f"   ✅ Successful: {successful_uploads}")
        logger.info(f"   ❌ Failed: {failed_uploads}")
        logger.info(f"   📁 Total files: {len(csv_files)}")
        
        if successful_uploads > 0:
            logger.info(f"\n🔍 View your data at:")
            logger.info(f"   https://console.cloud.google.com/bigquery?project={self.project_id}")
    
    def list_tables(self):
        """List all tables in the dataset"""
        try:
            dataset_ref = self.client.dataset(self.dataset_id)
            tables = list(self.client.list_tables(dataset_ref))
            
            if tables:
                logger.info(f"📋 Tables in dataset '{self.dataset_id}':")
                for table in tables:
                    table_info = self.client.get_table(table.reference)
                    logger.info(f"   • {table.table_id}: {table_info.num_rows:,} rows")
            else:
                logger.info(f"📋 No tables found in dataset '{self.dataset_id}'")
                
        except Exception as e:
            logger.error(f"❌ Error listing tables: {e}")


def main():
    """
    Main function - Configure your settings here
    """
    
    # ==========================================================================
    # 🔧 CONFIGURATION - UPDATE THESE VALUES
    # ==========================================================================
    
    PROJECT_ID = "data-eye-471403-d2"  # Replace with your Google Cloud Project ID
    DATASET_ID = "wind_analysis_NorthAmericans"    # Replace with your desired dataset name
    CREDENTIALS_PATH = "/home/dgoscbr/credentials/bigquery-key.json"  # Replace with path to your service account JSON
    
    CSV_DIRECTORY = "."  # Current directory, or specify path like "/path/to/your/csvs"
    CSV_PATTERN = "*.csv"  # Pattern to match CSV files
    
    # Upload behavior: 'replace' (overwrite), 'append' (add to existing), 'fail' (error if exists)
    IF_EXISTS = 'replace'
    
    # ==========================================================================
    
    # Validate configuration
    if PROJECT_ID == "your-project-id":
        logger.error("❌ Please update PROJECT_ID in the configuration section")
        return
    
    if CREDENTIALS_PATH == "path/to/your/credentials.json":
        logger.error("❌ Please update CREDENTIALS_PATH in the configuration section")
        return
    
    # Initialize uploader
    try:
        uploader = CSVToBigQueryUploader(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            credentials_path=CREDENTIALS_PATH
        )
        
        # Upload CSV files
        uploader.upload_multiple_csvs(
            csv_directory=CSV_DIRECTORY,
            pattern=CSV_PATTERN,
            if_exists=IF_EXISTS
        )
        
        # List final tables
        uploader.list_tables()
        
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")


if __name__ == "__main__":
    main()