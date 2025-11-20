# SQL Optimizer v4

A modular Python toolkit to analyze, optimize, and monitor SQL queries across multiple databases and SQL files. It supports PostgreSQL, MySQL, Microsoft SQL Server, and Snowflake, and includes advanced features like execution plan analysis, index recommendations, query complexity scoring, caching and parallelization analysis, cost estimation, regression testing, workload profiling, and Data Lake–focused optimizations.

# What’s New
- Batch `.sql` file analysis via `QueryOptimizationSuite.batch_analyze_file` with consolidated JSON reporting.
- Data Lake optimization checks (Snowflake and lakehouse patterns): CTAS, COPY INTO, external stages, columnar formats (PARQUET/ORC/DELTA), partition pruning, clustering, result caching, and guidance on LIMIT/QUALIFY for wide aggregations.

# Requirements
- Python 3.9+
- Packages:
  - `sqlparse`
  - Database drivers (install what you need):
    - PostgreSQL: `psycopg2` (or `psycopg2-binary`)
    - MySQL: `mysql-connector-python` (or `pymysql`)
    - SQL Server: `pyodbc`
    - Snowflake: `snowflake-connector-python`
- Optional tools for performance testing and monitoring are handled within the suite; no extra packages required beyond the above.

Install example:
```
pip install sqlparse psycopg2-binary mysql-connector-python pyodbc snowflake-connector-python
```

# Core Modules
- `SQLOptimizer` — detects optimization issues, suggests fixes, and aggregates a report.
- `DatabaseConnector` — manages connections, execution plans, table stats, slow query analysis, and real-time monitoring.
- `IndexRecommender` / `AutomatedIndexCreator` — generates and applies index recommendations.
- `QueryCachingAnalyzer` — identifies caching opportunities and strategies.
- `ParallelExecutionAnalyzer` — finds parallelization opportunities from query structure and plans.
- `CostEstimator` — estimates relative cost (includes Snowflake credits guidance).
- `RegressionTester` — runs repeatable A/B tests for original vs optimized queries.
- `WorkloadAnalyzer` — profiles a stream of queries for patterns and hotspots.
- `QueryOptimizationSuite` — orchestrates the above for end-to-end workflows and now adds `.sql` batch file analysis.

# Supported Databases
- `postgresql`, `mysql`, `sqlserver`, `snowflake`.

# Usage

## 1) Analyze one query (without DB connection)
```python
from sql_optimizer_v3_MLops import QueryOptimizationSuite, DatabaseType

suite = QueryOptimizationSuite(DatabaseType.POSTGRESQL)
result = suite.analyze_and_optimize(
    "SELECT * FROM orders WHERE order_date >= '2024-01-01'",
    auto_fix=True,
    get_execution_plan=False
)

print(result["analysis"].generate_report(result["issues"]))
print("Optimized:", result["optimized_query"])  # may be None if no auto-fix applicable
```

## 2) Connect to a database and analyze with execution plan
```python
from sql_optimizer_v3_MLops import QueryOptimizationSuite, DatabaseType

suite = QueryOptimizationSuite(
    DatabaseType.POSTGRESQL,
    {"host": "localhost", "database": "mydb", "user": "postgres", "password": "password"}
)
suite.connector.connect()

result = suite.analyze_and_optimize(
    "SELECT u.*, o.* FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'shipped'",
    auto_fix=True,
    get_execution_plan=True
)

print("Indexes:", result["index_recommendations"])
print("Caching:", result["caching_recommendations"]) 
print("Parallel:", result["parallel_execution"]) 
print("Cost:", result["cost_estimate"]) 

# Optionally test performance improvement if an optimized query is available
if result["optimized_query"]:
    test = suite.test_optimization(
        original_query=result["analysis"].original_query,
        optimized_query=result["optimized_query"],
        iterations=5
    )
    print("Speedup:", test["comparison"]["speedup_factor"], "x")
```

## 3) Batch analyze a `.sql` file
```python
from sql_optimizer_v3_MLops import QueryOptimizationSuite, DatabaseType

suite = QueryOptimizationSuite(DatabaseType.SNOWFLAKE)
results = suite.batch_analyze_file(
    "migrations.sql",
    auto_fix=True,
    output_file="comprehensive_analysis.json"  # optional
)

for r in results:
    print("Query:", r["normalized_query"][:120], "...")
    print("Complexity:", r["complexity_score"]) 
    print("Index Recommendations:", r["index_recommendations"]) 
    print("Caching:", r["caching_recommendations"]) 
    print("Parallelization:", r["parallel_execution"]) 
    print("Cost:", r["cost_estimate"]) 
```

# 4) Monitor slow queries and optimize
```python
from sql_optimizer_v3_MLops import QueryOptimizationSuite, DatabaseType

suite = QueryOptimizationSuite(
    DatabaseType.POSTGRESQL,
    {"host": "localhost", "database": "mydb", "user": "postgres", "password": "password"}
)
suite.connector.connect()

slow_queries = suite.connector.analyze_slow_queries(threshold_ms=1000, limit=10)
for sq in slow_queries:
    analysis = suite.analyze_and_optimize(sq["query"], auto_fix=True, get_execution_plan=True)
    print(analysis["analysis"].generate_report(analysis["issues"]))
```

# Data Lake Optimizations (Highlights)
- CTAS: recommend `CLUSTER BY` on high-selectivity columns and column pruning.
- External stages and COPY INTO: prefer PARQUET/ORC, apply partition filters, avoid `SELECT *`.
- Columnar formats: ensure column pruning and project minimal columns.
- Partition pruning: encourage alignment of predicates with clustering/partitioning columns.
- Wide aggregations without `LIMIT`: use `LIMIT` for exploration, pre-filter with selective `WHERE`.
- Snowflake specifics: suggest `CLUSTER BY` for large tables, leverage result caching when freshness allows.

# Feature Overview
- Query Optimization & Auto-Fix
- Live Database Connections (PostgreSQL, MySQL, SQL Server, Snowflake)
- Execution Plan Analysis
- Index Recommendations with SQL generation and optional auto-creation
- Query Complexity Scoring
- Caching Strategy Recommendations
- Parallel Execution Analysis
- Cost Estimation (including Snowflake credits guidance)
- Automated Regression Testing
- Performance Monitoring (slow query analysis and real-time metrics)
- Batch File Analysis (`.sql`)
- Comprehensive Reporting
- Query Pattern Detection (15+ patterns) and N+1 Detection
- Workload Analysis & Profiling and Optimization Opportunity Identification
- Unused Index Detection & Cleanup

# Tips and Best Practices
- Always start with `auto_fix=True` only in development or staging; review changes before production.
- For Snowflake, use `LIMIT` or strong filters when exploring large tables to control credits.
- Validate correctness via `RegressionTester` when applying optimizations that modify semantics.
- Keep indexes targeted: avoid overlapping redundant indexes and monitor actual usage.

# Troubleshooting
- Missing database driver: install the appropriate package listed in Requirements.
- Execution plan retrieval may vary by database permissions: ensure your user has EXPLAIN/SHOW PLAN rights.
- `pyodbc` requires a DSN or appropriate driver string for SQL Server; confirm driver installation.
- Snowflake connections need correct `account`, `user`, `password`, `warehouse`, `database`, and `schema` settings.

# Roadmap
- Optional CLI entry point for `--db` and `--file` workflows.
- Advanced plan-aware refactor suggestions and automated materialization strategies.