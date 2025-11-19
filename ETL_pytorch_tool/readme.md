Comprehensive ETL tool that demonstrates data extraction, transformation, and loading using open-source tools. This project will extract data from a source, transform it using PyTorch for ML-based transformations, and load it into both PostgreSQL and Snowflake.



# Key Features:

#1. Extract Layer
- CSV file extraction
- PostgreSQL database extraction
- API data extraction (simulated)

#2. Transform Layer with PyTorch
- Data Cleaning: Removes duplicates, handles missing values
- Feature Engineering: Creates time-based and aggregate features
- ML-based Anomaly Detection: Uses a PyTorch autoencoder neural network to detect anomalies in transaction data

#3. Load Layer
- PostgreSQL loading using SQLAlchemy
- Snowflake loading using native connector
- CSV file output

# How to Use:

1. Install dependencies:
```bash
pip install pandas numpy torch psycopg2-binary snowflake-connector-python sqlalchemy python-dotenv
```

2. Configure your connections:
   - Update the `config` dictionary with your database credentials
   - Uncomment the PostgreSQL or Snowflake destination blocks

3. Run the pipeline:
```bash
python etl_pipeline.py
```

# Architecture Highlights:

- Modular Design: Separate classes for Extract, Transform, and Load
- PyTorch Integration: Neural network for intelligent anomaly detection
- Logging: Comprehensive logging throughout the pipeline
- Flexible Configuration: Easy to add new sources and destinations
- Error Handling: Try-catch blocks for robust execution

The pipeline currently simulates API data and outputs to CSV by default, but you can easily enable PostgreSQL and Snowflake by uncommenting and configuring the connection parameters!



"""
ETL Pipeline Project
====================
Extract, Transform, Load pipeline using Python, PyTorch, PostgreSQL, and Snowflake

Requirements:
pip install pandas numpy torch psycopg2-binary snowflake-connector-python sqlalchemy python-dotenv
"""
