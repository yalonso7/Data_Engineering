import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import psycopg2
from sqlalchemy import create_engine
import snowflake.connector
from typing import Dict, List, Optional
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXTRACT: Data Extraction Layer
# ============================================================================

class DataExtractor:
    """Handles data extraction from various sources"""
    
    def extract_from_csv(self, filepath: str) -> pd.DataFrame:
        """Extract data from CSV file"""
        logger.info(f"Extracting data from CSV: {filepath}")
        return pd.read_csv(filepath)
    
    def extract_from_postgresql(self, conn_params: Dict, query: str) -> pd.DataFrame:
        """Extract data from PostgreSQL database"""
        logger.info("Extracting data from PostgreSQL")
        conn = psycopg2.connect(**conn_params)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def extract_from_api(self, url: str) -> pd.DataFrame:
        """Extract data from REST API (simulated)"""
        logger.info(f"Extracting data from API: {url}")
        # Simulated API data
        data = {
            'transaction_id': range(1000, 1500),
            'customer_id': np.random.randint(1, 100, 500),
            'amount': np.random.uniform(10, 1000, 500),
            'timestamp': pd.date_range(start='2024-01-01', periods=500, freq='H'),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 500)
        }
        return pd.DataFrame(data)


# ============================================================================
# TRANSFORM: Data Transformation Layer with PyTorch
# ============================================================================

class AnomalyDetector(nn.Module):
    """PyTorch-based autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int):
        super(AnomalyDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DataTransformer:
    """Handles data transformation including ML-based transformations"""
    
    def __init__(self):
        self.model = None
        self.scaler_params = None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning"""
        logger.info("Cleaning data")
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Handle categorical missing values
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""
        logger.info("Engineering features")
        df = df.copy()
        
        # Time-based features if timestamp exists
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Aggregate features if customer_id exists
        if 'customer_id' in df.columns and 'amount' in df.columns:
            customer_stats = df.groupby('customer_id')['amount'].agg([
                ('customer_avg_amount', 'mean'),
                ('customer_total_amount', 'sum'),
                ('customer_transaction_count', 'count')
            ]).reset_index()
            df = df.merge(customer_stats, on='customer_id', how='left')
        
        return df
    
    def detect_anomalies(self, df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """Detect anomalies using PyTorch autoencoder"""
        logger.info("Detecting anomalies with PyTorch")
        df = df.copy()
        
        # Select numeric features for anomaly detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['transaction_id', 'customer_id']]
        
        if not numeric_cols:
            logger.warning("No numeric columns found for anomaly detection")
            df['is_anomaly'] = 0
            df['anomaly_score'] = 0.0
            return df
        
        # Prepare data
        X = df[numeric_cols].values
        
        # Normalize data
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        X_normalized = (X - mean) / std
        self.scaler_params = {'mean': mean, 'std': std}
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_normalized)
        
        # Initialize and train autoencoder
        input_dim = X_tensor.shape[1]
        self.model = AnomalyDetector(input_dim)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, X_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
        
        # Calculate anomaly scores
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            anomaly_scores = mse.numpy()
        
        # Threshold-based anomaly detection
        threshold_value = np.mean(anomaly_scores) + threshold * np.std(anomaly_scores)
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = (anomaly_scores > threshold_value).astype(int)
        
        logger.info(f"Detected {df['is_anomaly'].sum()} anomalies out of {len(df)} records")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete transformation pipeline"""
        logger.info("Starting transformation pipeline")
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.detect_anomalies(df)
        logger.info("Transformation pipeline completed")
        return df


# ============================================================================
# LOAD: Data Loading Layer
# ============================================================================

class DataLoader:
    """Handles data loading to various destinations"""
    
    def load_to_postgresql(self, df: pd.DataFrame, conn_params: Dict, table_name: str):
        """Load data to PostgreSQL"""
        logger.info(f"Loading data to PostgreSQL table: {table_name}")
        
        engine = create_engine(
            f"postgresql://{conn_params['user']}:{conn_params['password']}@"
            f"{conn_params['host']}:{conn_params.get('port', 5432)}/{conn_params['database']}"
        )
        
        df.to_sql(table_name, engine, if_exists='replace', index=False, method='multi')
        logger.info(f"Successfully loaded {len(df)} records to PostgreSQL")
    
    def load_to_snowflake(self, df: pd.DataFrame, conn_params: Dict, table_name: str):
        """Load data to Snowflake"""
        logger.info(f"Loading data to Snowflake table: {table_name}")
        
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        
        # Create table if not exists (simplified schema)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join([f'{col} VARCHAR' if df[col].dtype == 'object' 
                       else f'{col} FLOAT' for col in df.columns])}
        )
        """
        cursor.execute(create_table_sql)
        
        # Insert data
        cols = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
        
        cursor.executemany(insert_sql, df.values.tolist())
        conn.commit()
        
        cursor.close()
        conn.close()
        logger.info(f"Successfully loaded {len(df)} records to Snowflake")
    
    def load_to_csv(self, df: pd.DataFrame, filepath: str):
        """Load data to CSV file"""
        logger.info(f"Loading data to CSV: {filepath}")
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully loaded {len(df)} records to CSV")


# ============================================================================
# ETL Pipeline Orchestrator
# ============================================================================

class ETLPipeline:
    """Main ETL pipeline orchestrator"""
    
    def __init__(self):
        self.extractor = DataExtractor()
        self.transformer = DataTransformer()
        self.loader = DataLoader()
    
    def run(self, config: Dict):
        """Execute the complete ETL pipeline"""
        logger.info("=" * 80)
        logger.info("Starting ETL Pipeline")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # EXTRACT
            logger.info("PHASE 1: EXTRACTION")
            if config['source']['type'] == 'api':
                df = self.extractor.extract_from_api(config['source']['url'])
            elif config['source']['type'] == 'csv':
                df = self.extractor.extract_from_csv(config['source']['filepath'])
            elif config['source']['type'] == 'postgresql':
                df = self.extractor.extract_from_postgresql(
                    config['source']['conn_params'],
                    config['source']['query']
                )
            else:
                raise ValueError(f"Unsupported source type: {config['source']['type']}")
            
            logger.info(f"Extracted {len(df)} records")
            
            # TRANSFORM
            logger.info("\nPHASE 2: TRANSFORMATION")
            df_transformed = self.transformer.transform(df)
            logger.info(f"Transformed data shape: {df_transformed.shape}")
            
            # LOAD
            logger.info("\nPHASE 3: LOADING")
            for destination in config['destinations']:
                if destination['type'] == 'postgresql':
                    self.loader.load_to_postgresql(
                        df_transformed,
                        destination['conn_params'],
                        destination['table_name']
                    )
                elif destination['type'] == 'snowflake':
                    self.loader.load_to_snowflake(
                        df_transformed,
                        destination['conn_params'],
                        destination['table_name']
                    )
                elif destination['type'] == 'csv':
                    self.loader.load_to_csv(
                        df_transformed,
                        destination['filepath']
                    )
            
            # Pipeline completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("=" * 80)
            logger.info(f"ETL Pipeline completed successfully in {duration:.2f} seconds")
            logger.info(f"Records processed: {len(df_transformed)}")
            logger.info(f"Anomalies detected: {df_transformed['is_anomaly'].sum()}")
            logger.info("=" * 80)
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {str(e)}")
            raise


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = {
        'source': {
            'type': 'api',  # Options: 'api', 'csv', 'postgresql'
            'url': 'https://api.example.com/transactions'
        },
        'destinations': [
            {
                'type': 'csv',
                'filepath': 'output_data.csv'
            },
            # Uncomment and configure for PostgreSQL
            # {
            #     'type': 'postgresql',
            #     'conn_params': {
            #         'host': 'localhost',
            #         'database': 'etl_db',
            #         'user': 'postgres',
            #         'password': 'your_password',
            #         'port': 5432
            #     },
            #     'table_name': 'transformed_transactions'
            # },
            # Uncomment and configure for Snowflake
            # {
            #     'type': 'snowflake',
            #     'conn_params': {
            #         'user': 'your_user',
            #         'password': 'your_password',
            #         'account': 'your_account',
            #         'warehouse': 'your_warehouse',
            #         'database': 'your_database',
            #         'schema': 'your_schema'
            #     },
            #     'table_name': 'transformed_transactions'
            # }
        ]
    }
    
    # Run ETL pipeline
    pipeline = ETLPipeline()
    result_df = pipeline.run(config)
    
    # Display results
    print("\nSample of transformed data:")
    print(result_df.head())
    print(f"\nData shape: {result_df.shape}")
    print(f"\nColumns: {result_df.columns.tolist()}")
