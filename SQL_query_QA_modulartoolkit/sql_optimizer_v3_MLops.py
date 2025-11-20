"""
SQL Optimization and Performance Monitoring Tool
Supports: PostgreSQL, MySQL, SQL Server, Snowflake

Features:
- Live database connections
- Auto-fix capabilities
- Query execution plan analysis
- Performance monitoring
- Query complexity scoring
- Index recommendations
- Statistics collection
- Query caching recommendations
- Parallel execution analysis
- Cost estimation
- Automated regression testing
- Query pattern detection
- Workload analysis
- Automated index creation
"""

import re
import sqlparse
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict, Counter


class DatabaseType(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLSERVER = "sqlserver"
    SNOWFLAKE = "snowflake"


class QueryPattern(Enum):
    """Common query patterns"""
    CRUD_INSERT = "crud_insert"
    CRUD_SELECT = "crud_select"
    CRUD_UPDATE = "crud_update"
    CRUD_DELETE = "crud_delete"
    ANALYTICS_AGGREGATION = "analytics_aggregation"
    ANALYTICS_WINDOW = "analytics_window"
    REPORTING_JOIN = "reporting_join"
    BATCH_OPERATION = "batch_operation"
    N_PLUS_1 = "n_plus_1"
    FULL_TABLE_SCAN = "full_table_scan"
    CARTESIAN_PRODUCT = "cartesian_product"
    SUBQUERY_HEAVY = "subquery_heavy"
    RECURSIVE_CTE = "recursive_cte"
    PIVOT_UNPIVOT = "pivot_unpivot"
    TIME_SERIES = "time_series"
    SEARCH_PATTERN = "search_pattern"


@dataclass
class OptimizationIssue:
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str
    description: str
    recommendation: str
    query_section: Optional[str] = None
    auto_fixable: bool = False
    optimized_query: Optional[str] = None
    estimated_impact: Optional[str] = None


@dataclass
class QueryMetrics:
    execution_time_ms: Optional[float] = None
    rows_examined: Optional[int] = None
    rows_returned: Optional[int] = None
    memory_used_mb: Optional[float] = None
    cpu_time_ms: Optional[float] = None
    io_cost: Optional[float] = None
    temp_tables: Optional[int] = None
    filesort: bool = False
    full_table_scan: bool = False


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    query_hash: str
    issues: List[OptimizationIssue]
    metrics: Optional[QueryMetrics] = None
    execution_plan: Optional[Dict] = None
    complexity_score: int = 0
    optimized_query: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    detected_patterns: List[QueryPattern] = field(default_factory=list)


@dataclass
class IndexRecommendation:
    table: str
    columns: List[str]
    index_type: str
    priority: str
    reason: str
    sql_create: str
    estimated_benefit: str
    supporting_queries: List[str] = field(default_factory=list)


class QueryPatternDetector:
    """Detect and classify query patterns"""
    
    def __init__(self):
        self.pattern_cache: Dict[str, List[QueryPattern]] = {}
    
    def detect_patterns(self, query: str, query_hash: str) -> List[QueryPattern]:
        """Detect all patterns in a query"""
        
        if query_hash in self.pattern_cache:
            return self.pattern_cache[query_hash]
        
        patterns = []
        query_upper = query.upper()
        
        # CRUD patterns
        if re.search(r'^\s*INSERT\s+INTO', query_upper):
            patterns.append(QueryPattern.CRUD_INSERT)
        elif re.search(r'^\s*SELECT', query_upper):
            patterns.append(QueryPattern.CRUD_SELECT)
        elif re.search(r'^\s*UPDATE', query_upper):
            patterns.append(QueryPattern.CRUD_UPDATE)
        elif re.search(r'^\s*DELETE', query_upper):
            patterns.append(QueryPattern.CRUD_DELETE)
        
        # Analytics patterns
        if re.search(r'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(', query_upper):
            patterns.append(QueryPattern.ANALYTICS_AGGREGATION)
        
        if re.search(r'\b(ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|OVER)\s*\(', query_upper):
            patterns.append(QueryPattern.ANALYTICS_WINDOW)
        
        # Reporting patterns
        join_count = len(re.findall(r'\bJOIN\b', query_upper))
        if join_count >= 3:
            patterns.append(QueryPattern.REPORTING_JOIN)
        
        # Problematic patterns
        if not re.search(r'\bWHERE\b', query_upper) and QueryPattern.CRUD_SELECT in patterns:
            patterns.append(QueryPattern.FULL_TABLE_SCAN)
        
        if re.search(r'\bCROSS\s+JOIN\b', query_upper):
            patterns.append(QueryPattern.CARTESIAN_PRODUCT)
        
        subquery_count = len(re.findall(r'\(\s*SELECT', query_upper))
        if subquery_count >= 3:
            patterns.append(QueryPattern.SUBQUERY_HEAVY)
        
        # Advanced patterns
        if re.search(r'\bWITH\s+RECURSIVE\b', query_upper):
            patterns.append(QueryPattern.RECURSIVE_CTE)
        
        if re.search(r'\bPIVOT\b|\bUNPIVOT\b', query_upper):
            patterns.append(QueryPattern.PIVOT_UNPIVOT)
        
        # Time series pattern
        if re.search(r'\b(DATE|TIMESTAMP|TIME).*\b(BETWEEN|>=|<=)', query_upper):
            patterns.append(QueryPattern.TIME_SERIES)
        
        # Search pattern
        if re.search(r'\bLIKE\s+[\'"]%', query_upper):
            patterns.append(QueryPattern.SEARCH_PATTERN)
        
        # Batch operations
        if re.search(r'\bINSERT.*VALUES.*,.*,', query_upper, re.DOTALL):
            patterns.append(QueryPattern.BATCH_OPERATION)
        
        self.pattern_cache[query_hash] = patterns
        return patterns
    
    def detect_n_plus_1(self, queries: List[str], time_window_sec: float = 1.0) -> Dict:
        """Detect N+1 query problem by analyzing query patterns"""
        
        # Normalize queries to detect similar patterns
        normalized_patterns = defaultdict(list)
        
        for i, query in enumerate(queries):
            # Replace numbers and strings with placeholders
            normalized = re.sub(r'\d+', '?', query)
            normalized = re.sub(r'\'[^\']*\'', '?', normalized)
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            normalized_patterns[normalized].append(i)
        
        # Find patterns that repeat many times
        n_plus_1_candidates = []
        
        for pattern, indices in normalized_patterns.items():
            if len(indices) >= 10:  # Repeated 10+ times
                n_plus_1_candidates.append({
                    'pattern': pattern,
                    'occurrences': len(indices),
                    'query_indices': indices,
                    'severity': 'critical' if len(indices) > 100 else 'high',
                    'recommendation': 'Use JOIN or IN clause instead of multiple queries'
                })
        
        return {
            'detected': len(n_plus_1_candidates) > 0,
            'candidates': n_plus_1_candidates
        }


class WorkloadAnalyzer:
    """Analyze query workload patterns and characteristics"""
    
    def __init__(self):
        self.query_history: List[Dict] = []
        self.pattern_stats: Dict[QueryPattern, Dict] = defaultdict(lambda: {
            'count': 0,
            'total_time_ms': 0,
            'avg_time_ms': 0,
            'queries': []
        })
    
    def add_query_execution(self, query: str, execution_time_ms: float, 
                           patterns: List[QueryPattern], timestamp: Optional[datetime] = None):
        """Record a query execution"""
        
        execution = {
            'query': query,
            'query_hash': hashlib.md5(query.encode()).hexdigest(),
            'execution_time_ms': execution_time_ms,
            'patterns': patterns,
            'timestamp': timestamp or datetime.now()
        }
        
        self.query_history.append(execution)
        
        # Update pattern statistics
        for pattern in patterns:
            stats = self.pattern_stats[pattern]
            stats['count'] += 1
            stats['total_time_ms'] += execution_time_ms
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['count']
            if len(stats['queries']) < 5:  # Keep sample queries
                stats['queries'].append(query[:200])
    
    def analyze_workload(self, time_period_hours: Optional[int] = None) -> Dict:
        """Analyze workload patterns"""
        
        if not self.query_history:
            return {'error': 'No query history available'}
        
        # Filter by time period if specified
        if time_period_hours:
            cutoff = datetime.now() - timedelta(hours=time_period_hours)
            filtered_history = [q for q in self.query_history if q['timestamp'] >= cutoff]
        else:
            filtered_history = self.query_history
        
        if not filtered_history:
            return {'error': 'No queries in specified time period'}
        
        # Calculate statistics
        total_queries = len(filtered_history)
        total_time = sum(q['execution_time_ms'] for q in filtered_history)
        avg_time = total_time / total_queries
        
        # Find slowest queries
        slowest = sorted(filtered_history, key=lambda x: x['execution_time_ms'], reverse=True)[:10]
        
        # Pattern distribution
        pattern_distribution = Counter()
        for query in filtered_history:
            for pattern in query['patterns']:
                pattern_distribution[pattern.value] += 1
        
        # Time-based analysis
        queries_by_hour = defaultdict(int)
        for query in filtered_history:
            hour = query['timestamp'].hour
            queries_by_hour[hour] += 1
        
        peak_hour = max(queries_by_hour.items(), key=lambda x: x[1])[0] if queries_by_hour else None
        
        # Query frequency analysis
        query_frequency = Counter(q['query_hash'] for q in filtered_history)
        most_frequent = query_frequency.most_common(10)
        
        # Calculate percentiles
        times = sorted([q['execution_time_ms'] for q in filtered_history])
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]
        
        return {
            'summary': {
                'total_queries': total_queries,
                'total_time_ms': total_time,
                'avg_time_ms': avg_time,
                'time_period_hours': time_period_hours,
                'queries_per_hour': total_queries / max(time_period_hours, 1) if time_period_hours else None
            },
            'percentiles': {
                'p50_ms': p50,
                'p95_ms': p95,
                'p99_ms': p99
            },
            'slowest_queries': [
                {
                    'query': q['query'][:200],
                    'time_ms': q['execution_time_ms'],
                    'patterns': [p.value for p in q['patterns']]
                }
                for q in slowest
            ],
            'pattern_distribution': dict(pattern_distribution),
            'peak_hour': peak_hour,
            'most_frequent_queries': [
                {
                    'query_hash': qh[:8],
                    'frequency': freq,
                    'percentage': (freq / total_queries) * 100
                }
                for qh, freq in most_frequent
            ],
            'pattern_performance': {
                pattern.value: {
                    'count': stats['count'],
                    'avg_time_ms': stats['avg_time_ms'],
                    'total_time_ms': stats['total_time_ms']
                }
                for pattern, stats in self.pattern_stats.items()
            }
        }
    
    def identify_optimization_opportunities(self) -> List[Dict]:
        """Identify top optimization opportunities based on workload"""
        
        opportunities = []
        
        # High-frequency slow queries
        query_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'queries': []})
        
        for query in self.query_history:
            qh = query['query_hash']
            query_stats[qh]['count'] += 1
            query_stats[qh]['total_time'] += query['execution_time_ms']
            if len(query_stats[qh]['queries']) < 1:
                query_stats[qh]['queries'].append(query['query'])
        
        # Calculate impact score (frequency × time)
        for qh, stats in query_stats.items():
            impact = stats['count'] * stats['total_time']
            avg_time = stats['total_time'] / stats['count']
            
            if impact > 10000:  # Arbitrary threshold
                opportunities.append({
                    'type': 'high_impact_query',
                    'query_hash': qh[:8],
                    'query': stats['queries'][0][:200],
                    'frequency': stats['count'],
                    'avg_time_ms': avg_time,
                    'total_time_ms': stats['total_time'],
                    'impact_score': impact,
                    'recommendation': 'Optimize this query - high frequency and cumulative time'
                })
        
        # Slow patterns
        for pattern, stats in self.pattern_stats.items():
            if stats['avg_time_ms'] > 1000 and stats['count'] > 10:
                opportunities.append({
                    'type': 'slow_pattern',
                    'pattern': pattern.value,
                    'count': stats['count'],
                    'avg_time_ms': stats['avg_time_ms'],
                    'recommendation': f'Optimize {pattern.value} pattern queries'
                })
        
        # Sort by impact
        opportunities.sort(key=lambda x: x.get('impact_score', x.get('avg_time_ms', 0)), reverse=True)
        
        return opportunities[:20]  # Top 20


class AutomatedIndexCreator:
    """Automatically create and manage indexes"""
    
    def __init__(self, db_type: DatabaseType):
        self.db_type = db_type
        self.recommended_indexes: List[IndexRecommendation] = []
        self.created_indexes: List[Dict] = []
    
    def analyze_workload_for_indexes(self, queries: List[str], 
                                     query_frequencies: Optional[Dict[str, int]] = None) -> List[IndexRecommendation]:
        """Analyze workload and recommend indexes"""
        
        # Track column usage across queries
        column_usage = defaultdict(lambda: {
            'where_count': 0,
            'join_count': 0,
            'order_count': 0,
            'group_count': 0,
            'queries': set()
        })
        
        table_columns = defaultdict(set)
        
        for query in queries:
            query_upper = query.upper()
            query_hash = hashlib.md5(query.encode()).hexdigest()
            freq = query_frequencies.get(query_hash, 1) if query_frequencies else 1
            
            # Extract WHERE columns
            where_matches = re.findall(r'WHERE\s+(?:(\w+)\.)?(\w+)\s*[=<>]', query_upper)
            for table, col in where_matches:
                key = f"{table or 'unknown'}.{col}"
                column_usage[key]['where_count'] += freq
                column_usage[key]['queries'].add(query_hash)
                if table:
                    table_columns[table].add(col)
            
            # Extract JOIN columns
            join_matches = re.findall(r'ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', query_upper)
            for t1, c1, t2, c2 in join_matches:
                for table, col in [(t1, c1), (t2, c2)]:
                    key = f"{table}.{col}"
                    column_usage[key]['join_count'] += freq
                    column_usage[key]['queries'].add(query_hash)
                    table_columns[table].add(col)
            
            # Extract ORDER BY columns
            order_match = re.search(r'ORDER\s+BY\s+((?:(?:\w+\.)?\w+(?:\s+(?:ASC|DESC))?\s*,?\s*)+)', query_upper)
            if order_match:
                order_cols = re.findall(r'(?:(\w+)\.)?(\w+)', order_match.group(1))
                for table, col in order_cols:
                    key = f"{table or 'unknown'}.{col}"
                    column_usage[key]['order_count'] += freq
                    column_usage[key]['queries'].add(query_hash)
                    if table:
                        table_columns[table].add(col)
            
            # Extract GROUP BY columns
            group_match = re.search(r'GROUP\s+BY\s+((?:(?:\w+\.)?\w+\s*,?\s*)+)', query_upper)
            if group_match:
                group_cols = re.findall(r'(?:(\w+)\.)?(\w+)', group_match.group(1))
                for table, col in group_cols:
                    key = f"{table or 'unknown'}.{col}"
                    column_usage[key]['group_count'] += freq
                    column_usage[key]['queries'].add(query_hash)
                    if table:
                        table_columns[table].add(col)
        
        # Generate recommendations
        recommendations = []
        
        for col_key, usage in column_usage.items():
            if '.' not in col_key:
                continue
            
            table, column = col_key.split('.')
            
            # Calculate priority score
            score = (
                usage['where_count'] * 10 +
                usage['join_count'] * 8 +
                usage['order_count'] * 3 +
                usage['group_count'] * 3
            )
            
            if score < 5:
                continue
            
            # Determine priority
            if score >= 50:
                priority = 'critical'
            elif score >= 20:
                priority = 'high'
            elif score >= 10:
                priority = 'medium'
            else:
                priority = 'low'
            
            # Determine index type
            if usage['where_count'] > 0 or usage['join_count'] > 0:
                index_type = 'B-tree'
                reason = f"Used in WHERE ({usage['where_count']}) and JOIN ({usage['join_count']})"
            elif usage['order_count'] > 0:
                index_type = 'B-tree'
                reason = f"Used in ORDER BY ({usage['order_count']})"
            else:
                index_type = 'B-tree'
                reason = f"Used in GROUP BY ({usage['group_count']})"
            
            # Generate SQL
            sql_create = self._generate_index_sql(table, [column], index_type)
            
            # Estimate benefit
            query_count = len(usage['queries'])
            if query_count > 50:
                benefit = f"High impact - affects {query_count} queries"
            elif query_count > 10:
                benefit = f"Medium impact - affects {query_count} queries"
            else:
                benefit = f"Low impact - affects {query_count} queries"
            
            recommendations.append(IndexRecommendation(
                table=table,
                columns=[column],
                index_type=index_type,
                priority=priority,
                reason=reason,
                sql_create=sql_create,
                estimated_benefit=benefit,
                supporting_queries=list(usage['queries'])[:5]
            ))
        
        # Generate composite index recommendations
        for table, columns in table_columns.items():
            if len(columns) >= 2:
                # Find commonly co-occurring columns
                col_list = list(columns)[:3]  # Top 3 columns
                composite_score = sum(
                    column_usage.get(f"{table}.{col}", {}).get('where_count', 0)
                    for col in col_list
                )
                
                if composite_score >= 20:
                    sql_create = self._generate_index_sql(table, col_list, 'B-tree')
                    
                    recommendations.append(IndexRecommendation(
                        table=table,
                        columns=col_list,
                        index_type='B-tree',
                        priority='medium',
                        reason=f"Composite index for multiple WHERE conditions",
                        sql_create=sql_create,
                        estimated_benefit="Improves multi-column queries",
                        supporting_queries=[]
                    ))
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order[x.priority])
        
        self.recommended_indexes = recommendations
        return recommendations
    
    def _generate_index_sql(self, table: str, columns: List[str], index_type: str) -> str:
        """Generate CREATE INDEX SQL"""
        
        index_name = f"idx_{table}_{'_'.join(columns)}"[:63]  # Max identifier length
        cols_str = ', '.join(columns)
        
        if self.db_type == DatabaseType.POSTGRESQL:
            if index_type.lower() == 'b-tree':
                return f"CREATE INDEX CONCURRENTLY {index_name} ON {table} ({cols_str});"
            return f"CREATE INDEX CONCURRENTLY {index_name} ON {table} USING {index_type} ({cols_str});"
        
        elif self.db_type == DatabaseType.MYSQL:
            return f"CREATE INDEX {index_name} ON {table} ({cols_str});"
        
        elif self.db_type == DatabaseType.SQLSERVER:
            return f"CREATE INDEX {index_name} ON {table} ({cols_str});"
        
        elif self.db_type == DatabaseType.SNOWFLAKE:
            return f"-- Note: Snowflake uses automatic clustering\n-- ALTER TABLE {table} CLUSTER BY ({cols_str});"
        
        return ""
    
    def create_indexes(self, connection, recommendations: Optional[List[IndexRecommendation]] = None,
                      dry_run: bool = True, max_indexes: int = 10) -> Dict:
        """Create recommended indexes"""
        
        recs = recommendations or self.recommended_indexes[:max_indexes]
        
        results = {
            'total_recommendations': len(recs),
            'created': [],
            'failed': [],
            'skipped': [],
            'dry_run': dry_run
        }
        
        if dry_run:
            print("DRY RUN MODE - No indexes will be created")
            results['dry_run_sql'] = [rec.sql_create for rec in recs]
            return results
        
        cursor = connection.cursor()
        
        for rec in recs:
            try:
                # Check if index already exists (database-specific)
                if self._index_exists(cursor, rec.table, rec.columns):
                    results['skipped'].append({
                        'table': rec.table,
                        'columns': rec.columns,
                        'reason': 'Index already exists'
                    })
                    continue
                
                # Create the index
                cursor.execute(rec.sql_create)
                connection.commit()
                
                results['created'].append({
                    'table': rec.table,
                    'columns': rec.columns,
                    'index_name': f"idx_{rec.table}_{'_'.join(rec.columns)}"[:63],
                    'sql': rec.sql_create
                })
                
                self.created_indexes.append({
                    'timestamp': datetime.now(),
                    'recommendation': rec
                })
                
                print(f"✓ Created index on {rec.table}({', '.join(rec.columns)})")
            
            except Exception as e:
                results['failed'].append({
                    'table': rec.table,
                    'columns': rec.columns,
                    'error': str(e)
                })
                print(f"✗ Failed to create index on {rec.table}: {str(e)}")
        
        cursor.close()
        return results
    
    def _index_exists(self, cursor, table: str, columns: List[str]) -> bool:
        """Check if an index already exists"""
        
        try:
            if self.db_type == DatabaseType.POSTGRESQL:
                query = """
                    SELECT 1 FROM pg_indexes 
                    WHERE tablename = %s 
                    AND indexdef LIKE %s
                """
                cursor.execute(query, (table, f"%({', '.join(columns)})%"))
                return cursor.fetchone() is not None
            
            elif self.db_type == DatabaseType.MYSQL:
                query = """
                    SELECT 1 FROM information_schema.statistics
                    WHERE table_name = %s AND column_name IN (%s)
                    GROUP BY index_name
                    HAVING COUNT(*) = %s
                """
                placeholders = ','.join(['%s'] * len(columns))
                cursor.execute(query, (table, *columns, len(columns)))
                return cursor.fetchone() is not None
            
            # For other databases, assume doesn't exist
            return False
        
        except:
            return False
    
    def drop_unused_indexes(self, connection, min_scans: int = 100, dry_run: bool = True) -> Dict:
        """Drop indexes that are not being used"""
        
        results = {
            'analyzed': [],
            'dropped': [],
            'kept': [],
            'dry_run': dry_run
        }
        
        if self.db_type != DatabaseType.POSTGRESQL:
            return {'error': 'Unused index detection only supported for PostgreSQL'}
        
        cursor = connection.cursor()
        
        # Query for unused indexes
        query = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_scan,
                pg_size_pretty(pg_relation_size(indexrelid)) as index_size
            FROM pg_stat_user_indexes
            WHERE idx_scan < %s
            AND indexname NOT LIKE '%%_pkey'
            ORDER BY pg_relation_size(indexrelid) DESC
        """
        
        cursor.execute(query, (min_scans,))
        unused_indexes = cursor.fetchall()
        
        for schema, table, index_name, scans, size in unused_indexes:
            results['analyzed'].append({
                'schema': schema,
                'table': table,
                'index': index_name,
                'scans': scans,
                'size': size
            })
            
            if not dry_run:
                try:
                    cursor.execute(f"DROP INDEX CONCURRENTLY {schema}.{index_name}")
                    connection.commit()
                    results['dropped'].append(index_name)
                    print(f"✓ Dropped unused index: {index_name}")
                except Exception as e:
                    print(f"✗ Failed to drop {index_name}: {str(e)}")
            else:
                print(f"Would drop: {index_name} (scans: {scans}, size: {size})")
        
        cursor.close()
        return results


class QueryAutoFixer:
    """Automatically fix common SQL issues"""
    
    def __init__(self, db_type: DatabaseType):
        self.db_type = db_type
    
    def fix_query(self, query: str, issues: List[OptimizationIssue]) -> str:
        """Apply automatic fixes to query"""
        fixed_query = query
        
        for issue in issues:
            if not issue.auto_fixable:
                continue
            
            if "= NULL" in issue.description or "NULL =" in issue.description:
                fixed_query = self._fix_null_comparison(fixed_query)
            
            elif "comma-separated tables" in issue.description:
                fixed_query = self._fix_implicit_joins(fixed_query)
            
            elif "UNION " in fixed_query and "UNION ALL" not in issue.description:
                if "Use UNION ALL" in issue.recommendation:
                    fixed_query = self._fix_union_to_union_all(fixed_query)
            
            elif "quoted numeric value" in issue.description:
                fixed_query = self._fix_quoted_numbers(fixed_query)
        
        return fixed_query
    
    def _fix_null_comparison(self, query: str) -> str:
        """Fix = NULL to IS NULL"""
        query = re.sub(r'(\w+)\s*=\s*NULL', r'\1 IS NULL', query, flags=re.IGNORECASE)
        query = re.sub(r'(\w+)\s*!=\s*NULL', r'\1 IS NOT NULL', query, flags=re.IGNORECASE)
        query = re.sub(r'(\w+)\s*<>\s*NULL', r'\1 IS NOT NULL', query, flags=re.IGNORECASE)
        return query
    
    def _fix_implicit_joins(self, query: str) -> str:
        """Convert comma joins to explicit JOIN syntax"""
        # Match FROM table1, table2 WHERE table1.id = table2.id
        pattern = r'FROM\s+(\w+)\s*,\s*(\w+)\s+WHERE\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            table1, table2, t1, col1, t2, col2 = match.groups()
            replacement = f'FROM {table1} JOIN {table2} ON {t1}.{col1} = {t2}.{col2} WHERE'
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        
        return query
    
    def _fix_union_to_union_all(self, query: str) -> str:
        """Convert UNION to UNION ALL"""
        return re.sub(r'\bUNION\s+(?!ALL)', 'UNION ALL ', query, flags=re.IGNORECASE)
    
    def _fix_quoted_numbers(self, query: str) -> str:
        """Remove quotes from numeric values"""
        return re.sub(r'=\s*["\'](\d+)["\']', r'= \1', query)


class IndexRecommender:
    """Recommend indexes based on query analysis"""
    
    def __init__(self, db_type: DatabaseType):
        self.db_type = db_type
        self.index_candidates: Dict[str, List[str]] = {}
    
    def analyze_query_for_indexes(self, query: str, execution_plan: Optional[Dict] = None) -> List[Dict]:
        """Recommend indexes based on query patterns"""
        recommendations = []
        
        # Extract table and column information
        where_conditions = self._extract_where_columns(query)
        join_conditions = self._extract_join_columns(query)
        order_by_cols = self._extract_order_by_columns(query)
        group_by_cols = self._extract_group_by_columns(query)
        
        # Recommend indexes for WHERE clauses
        for table, columns in where_conditions.items():
            recommendations.append({
                'type': 'WHERE clause optimization',
                'table': table,
                'columns': columns,
                'index_type': 'B-tree',
                'priority': 'high',
                'sql': self._generate_index_sql(table, columns, 'where')
            })
        
        # Recommend indexes for JOIN conditions
        for table, columns in join_conditions.items():
            recommendations.append({
                'type': 'JOIN optimization',
                'table': table,
                'columns': columns,
                'index_type': 'B-tree',
                'priority': 'high',
                'sql': self._generate_index_sql(table, columns, 'join')
            })
        
        # Recommend composite indexes for ORDER BY
        if order_by_cols:
            for table, columns in order_by_cols.items():
                recommendations.append({
                    'type': 'ORDER BY optimization',
                    'table': table,
                    'columns': columns,
                    'index_type': 'B-tree',
                    'priority': 'medium',
                    'sql': self._generate_index_sql(table, columns, 'order')
                })
        
        return recommendations
    
    def _extract_where_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in WHERE clause"""
        result = {}
        pattern = r'WHERE\s+(?:(\w+)\.)?(\w+)\s*[=<>]'
        matches = re.findall(pattern, query, re.IGNORECASE)
        
        for table, column in matches:
            table = table or 'unknown'
            if table not in result:
                result[table] = []
            if column not in result[table]:
                result[table].append(column)
        
        return result
    
    def _extract_join_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in JOIN conditions"""
        result = {}
        pattern = r'ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        matches = re.findall(pattern, query, re.IGNORECASE)
        
        for t1, c1, t2, c2 in matches:
            if t1 not in result:
                result[t1] = []
            if t2 not in result:
                result[t2] = []
            
            if c1 not in result[t1]:
                result[t1].append(c1)
            if c2 not in result[t2]:
                result[t2].append(c2)
        
        return result
    
    def _extract_order_by_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in ORDER BY"""
        result = {}
        pattern = r'ORDER\s+BY\s+((?:(?:\w+\.)?\w+(?:\s+(?:ASC|DESC))?\s*,?\s*)+)'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            order_clause = match.group(1)
            col_pattern = r'(?:(\w+)\.)?(\w+)'
            matches = re.findall(col_pattern, order_clause)
            
            for table, column in matches:
                table = table or 'unknown'
                if table not in result:
                    result[table] = []
                if column not in result[table]:
                    result[table].append(column)
        
        return result
    
    def _extract_group_by_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in GROUP BY"""
        result = {}
        pattern = r'GROUP\s+BY\s+((?:(?:\w+\.)?\w+\s*,?\s*)+)'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            group_clause = match.group(1)
            col_pattern = r'(?:(\w+)\.)?(\w+)'
            matches = re.findall(col_pattern, group_clause)
            
            for table, column in matches:
                table = table or 'unknown'
                if table not in result:
                    result[table] = []
                if column not in result[table]:
                    result[table].append(column)
        
        return result
    
    def _generate_index_sql(self, table: str, columns: List[str], purpose: str) -> str:
        """Generate CREATE INDEX statement"""
        index_name = f"idx_{table}_{'_'.join(columns[:3])}_{purpose}"
        cols_str = ', '.join(columns)
        
        if self.db_type == DatabaseType.POSTGRESQL:
            return f"CREATE INDEX {index_name} ON {table} ({cols_str});"
        elif self.db_type == DatabaseType.MYSQL:
            return f"CREATE INDEX {index_name} ON {table} ({cols_str});"
        elif self.db_type == DatabaseType.SQLSERVER:
            return f"CREATE INDEX {index_name} ON {table} ({cols_str});"
        elif self.db_type == DatabaseType.SNOWFLAKE:
            return f"-- Snowflake uses automatic clustering. Consider: ALTER TABLE {table} CLUSTER BY ({cols_str});"
        
        return ""


class QueryCachingAnalyzer:
    """Analyze queries for caching opportunities"""
    
    def __init__(self, db_type: DatabaseType):
        self.db_type = db_type
    
    def analyze_caching_potential(self, query: str, execution_time_ms: Optional[float] = None) -> Dict:
        """Determine if query is a good candidate for caching"""
        
        recommendations = {
            'cacheable': True,
            'cache_strategy': None,
            'reasons': [],
            'ttl_suggestion': None,
            'cache_key_columns': []
        }
        
        query_upper = query.upper()
        
        # Check for non-deterministic functions
        non_deterministic = [
            'NOW()', 'CURRENT_TIMESTAMP', 'RAND()', 'RANDOM()', 
            'UUID()', 'NEWID()', 'GETDATE()', 'CURRENT_DATE'
        ]
        
        for func in non_deterministic:
            if func in query_upper:
                recommendations['cacheable'] = False
                recommendations['reasons'].append(f'Contains non-deterministic function: {func}')
        
        # Check for write operations
        if re.search(r'\b(INSERT|UPDATE|DELETE|MERGE)\b', query_upper):
            recommendations['cacheable'] = False
            recommendations['reasons'].append('Write operation - not cacheable')
            return recommendations
        
        # Analyze query characteristics
        has_where = bool(re.search(r'\bWHERE\b', query_upper))
        has_joins = bool(re.search(r'\bJOIN\b', query_upper))
        has_aggregation = bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP BY)\b', query_upper))
        
        # Determine cache strategy
        if has_aggregation and not has_where:
            recommendations['cache_strategy'] = 'full_result'
            recommendations['ttl_suggestion'] = 3600  # 1 hour
            recommendations['reasons'].append('Aggregation without WHERE - cache full result')
        
        elif has_where:
            # Extract WHERE columns for parameterized caching
            where_cols = re.findall(r'WHERE\s+(?:\w+\.)?(\w+)', query_upper)
            recommendations['cache_strategy'] = 'parameterized'
            recommendations['cache_key_columns'] = list(set(where_cols))
            recommendations['ttl_suggestion'] = 300  # 5 minutes
            recommendations['reasons'].append('Parameterized query - cache by key columns')
        
        elif has_joins:
            recommendations['cache_strategy'] = 'materialized_view'
            recommendations['reasons'].append('Complex joins - consider materialized view')
        
        else:
            recommendations['cache_strategy'] = 'full_result'
            recommendations['ttl_suggestion'] = 1800  # 30 minutes
            recommendations['reasons'].append('Simple query - cache full result')
        
        # Consider execution time
        if execution_time_ms and execution_time_ms > 1000:
            recommendations['reasons'].append(f'Slow query ({execution_time_ms}ms) - high caching benefit')
        
        # Database-specific recommendations
        if self.db_type == DatabaseType.POSTGRESQL:
            if has_aggregation:
                recommendations['implementation'] = 'CREATE MATERIALIZED VIEW or application-level cache (Redis)'
        elif self.db_type == DatabaseType.MYSQL:
            recommendations['implementation'] = 'Query cache (deprecated in 8.0+) or application-level cache'
        elif self.db_type == DatabaseType.SNOWFLAKE:
            recommendations['implementation'] = 'Snowflake automatic result caching (24 hours)'
        
        return recommendations


class ParallelExecutionAnalyzer:
    """Analyze queries for parallel execution opportunities"""
    
    def __init__(self, db_type: DatabaseType):
        self.db_type = db_type
    
    def analyze_parallelization(self, query: str, execution_plan: Optional[Dict] = None) -> Dict:
        """Analyze if query can benefit from parallel execution"""
        
        analysis = {
            'parallelizable': False,
            'parallel_opportunities': [],
            'bottlenecks': [],
            'recommendations': [],
            'estimated_speedup': None
        }
        
        query_upper = query.upper()
        
        # Check for large table scans
        if re.search(r'\bFROM\s+\w+', query_upper) and not re.search(r'\bWHERE\b', query_upper):
            analysis['parallelizable'] = True
            analysis['parallel_opportunities'].append('Full table scan - can be parallelized')
        
        # Check for aggregations
        if re.search(r'\bGROUP BY\b', query_upper):
            analysis['parallelizable'] = True
            analysis['parallel_opportunities'].append('GROUP BY can use parallel aggregation')
        
        # Check for sorts
        if re.search(r'\bORDER BY\b', query_upper):
            analysis['parallel_opportunities'].append('ORDER BY can use parallel sort')
        
        # Check for multiple table scans (UNION)
        union_count = len(re.findall(r'\bUNION\b', query_upper))
        if union_count > 0:
            analysis['parallelizable'] = True
            analysis['parallel_opportunities'].append(f'{union_count} UNION branches can execute in parallel')
            analysis['estimated_speedup'] = f'{union_count}x potential speedup'
        
        # Check for subqueries that can be parallelized
        subquery_pattern = r'\(\s*SELECT.*?FROM.*?\)'
        subqueries = re.findall(subquery_pattern, query, re.IGNORECASE | re.DOTALL)
        if len(subqueries) >= 2:
            analysis['parallelizable'] = True
            analysis['parallel_opportunities'].append(f'{len(subqueries)} independent subqueries can execute in parallel')
        
        # Identify bottlenecks
        if re.search(r'\bDISTINCT\b', query_upper):
            analysis['bottlenecks'].append('DISTINCT requires serialization point')
        
        if re.search(r'\bORDER BY.*LIMIT\b', query_upper):
            analysis['bottlenecks'].append('ORDER BY + LIMIT may limit parallelization benefit')
        
        # Database-specific recommendations
        if self.db_type == DatabaseType.POSTGRESQL:
            analysis['recommendations'].append('Set max_parallel_workers_per_gather (default: 2)')
            analysis['recommendations'].append('Ensure parallel_setup_cost and parallel_tuple_cost are tuned')
            if analysis['parallelizable']:
                analysis['recommendations'].append('Use EXPLAIN (ANALYZE, VERBOSE) to verify parallel plan')
        
        elif self.db_type == DatabaseType.MYSQL:
            analysis['recommendations'].append('MySQL 8.0+ supports parallel index scans')
            analysis['recommendations'].append('Consider partitioning large tables')
        
        elif self.db_type == DatabaseType.SQLSERVER:
            analysis['recommendations'].append('SQL Server uses parallel execution automatically')
            analysis['recommendations'].append('Check MAXDOP settings and cost threshold')
        
        elif self.db_type == DatabaseType.SNOWFLAKE:
            analysis['recommendations'].append('Snowflake automatically parallelizes queries')
            analysis['recommendations'].append('Use larger warehouse size for more parallelism')
        
        return analysis


class CostEstimator:
    """Estimate query execution costs"""
    
    def __init__(self, db_type: DatabaseType):
        self.db_type = db_type
        
        # Cost factors (arbitrary units for comparison)
        self.costs = {
            'seq_scan': 100,
            'index_scan': 10,
            'index_seek': 1,
            'nested_loop': 50,
            'hash_join': 20,
            'merge_join': 15,
            'sort': 30,
            'aggregate': 25,
            'subquery': 40
        }
    
    def estimate_cost(self, query: str, table_sizes: Optional[Dict[str, int]] = None) -> Dict:
        """Estimate relative query cost"""
        
        cost_breakdown = {
            'total_cost': 0,
            'components': {},
            'cost_level': 'low',  # low, medium, high, very_high
            'primary_cost_drivers': [],
            'optimization_priority': []
        }
        
        query_upper = query.upper()
        table_sizes = table_sizes or {}
        
        # Estimate scan costs
        if not re.search(r'\bWHERE\b', query_upper):
            # Full table scan
            tables = re.findall(r'FROM\s+(\w+)', query_upper)
            for table in tables:
                size = table_sizes.get(table, 10000)  # Default estimate
                scan_cost = self.costs['seq_scan'] * (size / 1000)
                cost_breakdown['components'][f'{table}_scan'] = scan_cost
                cost_breakdown['total_cost'] += scan_cost
            
            cost_breakdown['primary_cost_drivers'].append('Full table scans')
        else:
            # Assume index usage
            tables = re.findall(r'FROM\s+(\w+)', query_upper)
            for table in tables:
                size = table_sizes.get(table, 10000)
                index_cost = self.costs['index_scan'] * (size / 10000)
                cost_breakdown['components'][f'{table}_index'] = index_cost
                cost_breakdown['total_cost'] += index_cost
        
        # Join costs
        join_count = len(re.findall(r'\bJOIN\b', query_upper))
        if join_count > 0:
            join_cost = self.costs['nested_loop'] * join_count
            cost_breakdown['components']['joins'] = join_cost
            cost_breakdown['total_cost'] += join_cost
            
            if join_count > 3:
                cost_breakdown['primary_cost_drivers'].append(f'Multiple joins ({join_count})')
        
        # Subquery costs
        subquery_count = len(re.findall(r'\(\s*SELECT', query_upper))
        if subquery_count > 0:
            subquery_cost = self.costs['subquery'] * subquery_count
            cost_breakdown['components']['subqueries'] = subquery_cost
            cost_breakdown['total_cost'] += subquery_cost
            
            if subquery_count > 2:
                cost_breakdown['primary_cost_drivers'].append(f'Multiple subqueries ({subquery_count})')
        
        # Aggregation costs
        if re.search(r'\bGROUP BY\b', query_upper):
            agg_cost = self.costs['aggregate']
            cost_breakdown['components']['aggregation'] = agg_cost
            cost_breakdown['total_cost'] += agg_cost
        
        # Sort costs
        if re.search(r'\bORDER BY\b', query_upper):
            sort_cost = self.costs['sort']
            
            # Higher cost for sorting large result sets
            if re.search(r'\bDISTINCT\b', query_upper):
                sort_cost *= 1.5
            
            cost_breakdown['components']['sort'] = sort_cost
            cost_breakdown['total_cost'] += sort_cost
        
        # Determine cost level
        total = cost_breakdown['total_cost']
        if total < 100:
            cost_breakdown['cost_level'] = 'low'
        elif total < 500:
            cost_breakdown['cost_level'] = 'medium'
        elif total < 1000:
            cost_breakdown['cost_level'] = 'high'
        else:
            cost_breakdown['cost_level'] = 'very_high'
        
        # Prioritize optimizations
        sorted_components = sorted(
            cost_breakdown['components'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for component, cost in sorted_components[:3]:
            if cost > 50:
                cost_breakdown['optimization_priority'].append(
                    f"Optimize {component} (cost: {cost:.1f})"
                )
        
        # Snowflake-specific cost estimation
        if self.db_type == DatabaseType.SNOWFLAKE:
            warehouse_cost = self._estimate_snowflake_cost(cost_breakdown['total_cost'])
            cost_breakdown['estimated_credits'] = warehouse_cost
        
        return cost_breakdown
    
    def _estimate_snowflake_cost(self, relative_cost: float) -> Dict:
        """Estimate Snowflake credit consumption"""
        
        # Rough estimates based on warehouse size
        warehouse_costs = {
            'X-Small': 1,
            'Small': 2,
            'Medium': 4,
            'Large': 8,
            'X-Large': 16
        }
        
        # Estimate execution time based on cost
        estimated_seconds = relative_cost / 10
        
        estimates = {}
        for size, credits_per_hour in warehouse_costs.items():
            credits = (estimated_seconds / 3600) * credits_per_hour
            estimates[size] = round(credits, 4)
        
        return estimates


class RegressionTester:
    """Automated regression testing for query optimizations"""
    
    def __init__(self, db_type: DatabaseType):
        self.db_type = db_type
        self.test_results: List[Dict] = []
    
    def create_test_suite(self, queries: List[str], baseline_metrics: Optional[List[Dict]] = None) -> Dict:
        """Create a test suite for queries"""
        
        test_suite = {
            'created_at': datetime.now(),
            'db_type': self.db_type.value,
            'test_cases': [],
            'baseline_metrics': baseline_metrics or []
        }
        
        for i, query in enumerate(queries):
            test_case = {
                'id': f'test_{i+1}',
                'query': query,
                'query_hash': hashlib.md5(query.encode()).hexdigest(),
                'expected_behavior': {
                    'should_execute': True,
                    'max_execution_time_ms': None,
                    'expected_row_count': None,
                    'expected_columns': None
                }
            }
            
            if baseline_metrics and i < len(baseline_metrics):
                baseline = baseline_metrics[i]
                test_case['expected_behavior'].update({
                    'max_execution_time_ms': baseline.get('execution_time_ms', 0) * 1.2,  # 20% tolerance
                    'expected_row_count': baseline.get('row_count'),
                })
            
            test_suite['test_cases'].append(test_case)
        
        return test_suite
    
    def run_regression_test(self, connection, original_query: str, 
                           optimized_query: str, iterations: int = 5) -> Dict:
        """Run regression test comparing original vs optimized query"""
        
        results = {
            'test_id': hashlib.md5(original_query.encode()).hexdigest()[:8],
            'timestamp': datetime.now(),
            'original_metrics': [],
            'optimized_metrics': [],
            'comparison': {},
            'passed': False,
            'issues': []
        }
        
        # Test original query
        for i in range(iterations):
            try:
                metrics = self._execute_and_measure(connection, original_query)
                results['original_metrics'].append(metrics)
            except Exception as e:
                results['issues'].append(f'Original query error: {str(e)}')
        
        # Test optimized query
        for i in range(iterations):
            try:
                metrics = self._execute_and_measure(connection, optimized_query)
                results['optimized_metrics'].append(metrics)
            except Exception as e:
                results['issues'].append(f'Optimized query error: {str(e)}')
        
        # Compare results
        if results['original_metrics'] and results['optimized_metrics']:
            results['comparison'] = self._compare_metrics(
                results['original_metrics'],
                results['optimized_metrics']
            )
            
            # Validate correctness
            correctness = self._validate_correctness(
                results['original_metrics'][0],
                results['optimized_metrics'][0]
            )
            
            results['passed'] = correctness['same_results'] and \
                               results['comparison']['performance_improvement'] >= 0
            
            if not correctness['same_results']:
                results['issues'].append('Results differ between original and optimized queries')
        
        self.test_results.append(results)
        return results
    
    def _execute_and_measure(self, connection, query: str) -> Dict:
        """Execute query and measure performance"""
        import time
        
        cursor = connection.cursor()
        
        start_time = time.time()
        cursor.execute(query)
        results = cursor.fetchall()
        end_time = time.time()
        
        metrics = {
            'execution_time_ms': (end_time - start_time) * 1000,
            'row_count': len(results),
            'column_count': len(cursor.description) if cursor.description else 0,
            'result_hash': hashlib.md5(str(results).encode()).hexdigest()
        }
        
        cursor.close()
        return metrics
    
    def _compare_metrics(self, original_metrics: List[Dict], 
                        optimized_metrics: List[Dict]) -> Dict:
        """Compare performance metrics"""
        
        avg_original_time = sum(m['execution_time_ms'] for m in original_metrics) / len(original_metrics)
        avg_optimized_time = sum(m['execution_time_ms'] for m in optimized_metrics) / len(optimized_metrics)
        
        improvement = ((avg_original_time - avg_optimized_time) / avg_original_time) * 100
        
        return {
            'avg_original_time_ms': avg_original_time,
            'avg_optimized_time_ms': avg_optimized_time,
            'performance_improvement': improvement,
            'speedup_factor': avg_original_time / avg_optimized_time if avg_optimized_time > 0 else 0,
            'verdict': 'faster' if improvement > 5 else 'slower' if improvement < -5 else 'similar'
        }
    
    def _validate_correctness(self, original_metrics: Dict, 
                             optimized_metrics: Dict) -> Dict:
        """Validate that results are identical"""
        
        return {
            'same_results': original_metrics['result_hash'] == optimized_metrics['result_hash'],
            'same_row_count': original_metrics['row_count'] == optimized_metrics['row_count'],
            'same_columns': original_metrics['column_count'] == optimized_metrics['column_count']
        }
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        
        if not self.test_results:
            return "No test results available"
        
        report = []
        report.append("=" * 80)
        report.append("REGRESSION TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total tests: {len(self.test_results)}")
        report.append("")
        
        passed = sum(1 for r in self.test_results if r['passed'])
        failed = len(self.test_results) - passed
        
        report.append(f"Passed: {passed}")
        report.append(f"Failed: {failed}")
        report.append("")
        
        for i, result in enumerate(self.test_results, 1):
            report.append(f"\nTest #{i} - {result['test_id']}")
            report.append("-" * 80)
            report.append(f"Status: {'✓ PASSED' if result['passed'] else '✗ FAILED'}")
            
            if result.get('comparison'):
                comp = result['comparison']
                report.append(f"Original avg time: {comp['avg_original_time_ms']:.2f}ms")
                report.append(f"Optimized avg time: {comp['avg_optimized_time_ms']:.2f}ms")
                report.append(f"Improvement: {comp['performance_improvement']:.1f}%")
                report.append(f"Speedup: {comp['speedup_factor']:.2f}x")
                report.append(f"Verdict: {comp['verdict']}")
            
            if result['issues']:
                report.append("\nIssues:")
                for issue in result['issues']:
                    report.append(f"  - {issue}")
        
        return "\n".join(report)


class QueryComplexityAnalyzer:
    """Analyze and score query complexity"""
    
    def calculate_complexity(self, query: str) -> Tuple[int, Dict[str, int]]:
        """Calculate complexity score (0-100)"""
        scores = {
            'joins': 0,
            'subqueries': 0,
            'aggregations': 0,
            'unions': 0,
            'case_statements': 0,
            'functions': 0,
            'wildcards': 0
        }
        
        query_upper = query.upper()
        
        # Count joins (10 points each, max 30)
        join_count = len(re.findall(r'\bJOIN\b', query_upper))
        scores['joins'] = min(join_count * 10, 30)
        
        # Count subqueries (15 points each, max 30)
        subquery_count = len(re.findall(r'SELECT.*FROM.*\(.*SELECT', query_upper, re.DOTALL))
        scores['subqueries'] = min(subquery_count * 15, 30)
        
        # Count aggregations (5 points each, max 15)
        agg_count = len(re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP BY)\b', query_upper))
        scores['aggregations'] = min(agg_count * 5, 15)
        
        # Count unions (10 points each, max 15)
        union_count = len(re.findall(r'\bUNION\b', query_upper))
        scores['unions'] = min(union_count * 10, 15)
        
        # Count CASE statements (5 points each, max 10)
        case_count = len(re.findall(r'\bCASE\b', query_upper))
        scores['case_statements'] = min(case_count * 5, 10)
        
        # Count function calls (2 points each, max 10)
        func_count = len(re.findall(r'\b\w+\s*\(', query))
        scores['functions'] = min(func_count * 2, 10)
        
        # Penalize SELECT * (5 points)
        if 'SELECT *' in query_upper:
            scores['wildcards'] = 5
        
        total_score = sum(scores.values())
        return total_score, scores


class SQLOptimizer:
    def __init__(self, db_type: DatabaseType = DatabaseType.POSTGRESQL):
        self.db_type = db_type
        self.issues: List[OptimizationIssue] = []
        self.auto_fixer = QueryAutoFixer(db_type)
        self.index_recommender = IndexRecommender(db_type)
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.caching_analyzer = QueryCachingAnalyzer(db_type)
        self.parallel_analyzer = ParallelExecutionAnalyzer(db_type)
        self.cost_estimator = CostEstimator(db_type)
        self.pattern_detector = QueryPatternDetector()
        
    def analyze_query_comprehensive(self, query: str, auto_fix: bool = False) -> QueryAnalysis:
        """Comprehensive query analysis with all features"""
        # Normalize query for comparison
        normalized = self._normalize_query(query)
        query_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        # Detect patterns
        detected_patterns = self.pattern_detector.detect_patterns(query, query_hash)
        
        # Analyze for issues
        issues = self.analyze_query(query)
        
        # Calculate complexity
        complexity_score, complexity_breakdown = self.complexity_analyzer.calculate_complexity(query)
        
        # Auto-fix if requested
        optimized_query = None
        if auto_fix and any(issue.auto_fixable for issue in issues):
            optimized_query = self.auto_fixer.fix_query(query, issues)
        
        # Get index recommendations
        index_recommendations = self.index_recommender.analyze_query_for_indexes(query)
        
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            query_hash=query_hash,
            issues=issues,
            complexity_score=complexity_score,
            optimized_query=optimized_query,
            detected_patterns=detected_patterns
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison (remove whitespace, lowercase)"""
        normalized = re.sub(r'\s+', ' ', query).strip().lower()
        return normalized
    
    def analyze_query(self, query: str) -> List[OptimizationIssue]:
        """Analyze a single SQL query"""
        self.issues = []
        
        if not query or not query.strip():
            return self.issues
        
        # Parse the query
        parsed = sqlparse.parse(query)
        if not parsed:
            return self.issues
        
        statement = parsed[0]
        query_upper = query.upper()
        
        # Run all analysis checks
        self._check_select_star(query, statement)
        self._check_missing_where(query, statement)
        self._check_non_sargable(query)
        self._check_implicit_conversions(query)
        self._check_or_conditions(query)
        self._check_subquery_optimization(query, statement)
        self._check_distinct_usage(query)
        self._check_join_types(query)
        self._check_index_hints(query)
        self._check_like_patterns(query)
        self._check_union_vs_union_all(query)
        self._check_aggregation_issues(query)
        self._check_null_checks(query)
        self._check_nested_loops(query)
        self._check_redundant_conditions(query)
        self._check_database_specific(query)
        self._check_data_lake_optimizations(query)
        
        return self.issues
    
    def _split_queries(self, content: str) -> List[str]:
        """Split SQL file into individual queries"""
        statements = sqlparse.split(content)
        return [s.strip() for s in statements if s.strip()]
    
    def _check_select_star(self, query: str, statement):
        """Check for SELECT * usage"""
        query_upper = query.upper()
        if re.search(r'\bSELECT\s+\*', query_upper):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Performance',
                description='SELECT * retrieves all columns, which can impact performance',
                recommendation='Explicitly specify only the columns you need',
                query_section='SELECT *',
                auto_fixable=False,
                estimated_impact='10-30% reduction in I/O for tables with many columns'
            ))
    
    def _check_missing_where(self, query: str, statement):
        """Check for SELECT/UPDATE/DELETE without WHERE clause"""
        query_upper = query.upper()
        
        if re.search(r'\b(UPDATE|DELETE)\b', query_upper):
            if not re.search(r'\bWHERE\b', query_upper):
                self.issues.append(OptimizationIssue(
                    severity='critical',
                    category='Safety',
                    description='UPDATE/DELETE without WHERE clause affects all rows',
                    recommendation='Always use WHERE clause to limit scope',
                    auto_fixable=False,
                    estimated_impact='Could modify entire table'
                ))
    
    def _check_non_sargable(self, query: str):
        """Check for non-SARGable predicates that prevent index usage"""
        patterns = [
            (r'\bWHERE\s+\w+\s*[\+\-\*\/]\s*\d+', 'Arithmetic operation on indexed column'),
            (r'\bWHERE\s+UPPER\(', 'Function on indexed column (UPPER)'),
            (r'\bWHERE\s+LOWER\(', 'Function on indexed column (LOWER)'),
            (r'\bWHERE\s+SUBSTRING\(', 'Function on indexed column (SUBSTRING)'),
            (r'\bWHERE\s+YEAR\(', 'Function on indexed column (YEAR)'),
            (r'\bWHERE\s+CAST\(', 'Type conversion on indexed column'),
            (r'\bWHERE\s+CONVERT\(', 'Type conversion on indexed column'),
        ]
        
        for pattern, desc in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                self.issues.append(OptimizationIssue(
                    severity='high',
                    category='Performance',
                    description=f'Non-SARGable predicate: {desc}',
                    recommendation='Rewrite query to avoid functions on indexed columns',
                    auto_fixable=False,
                    estimated_impact='Can cause full table scan instead of index seek'
                ))
    
    def _check_implicit_conversions(self, query: str):
        """Check for potential implicit type conversions"""
        if re.search(r'=\s*["\'][0-9]+["\']', query):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Performance',
                description='Potential implicit type conversion with quoted numeric value',
                recommendation='Use unquoted numbers for numeric columns',
                auto_fixable=True,
                estimated_impact='May prevent index usage'
            ))
    
    def _check_or_conditions(self, query: str):
        """Check for OR conditions that might benefit from UNION"""
        or_count = len(re.findall(r'\bOR\b', query, re.IGNORECASE))
        
        if or_count >= 3:
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Performance',
                description=f'Multiple OR conditions ({or_count}) may prevent index usage',
                recommendation='Consider rewriting as UNION ALL or using IN clause if applicable',
                auto_fixable=False,
                estimated_impact='Could improve index utilization'
            ))
    
    def _check_subquery_optimization(self, query: str, statement):
        """Check for subqueries that could be optimized"""
        if re.search(r'\bIN\s*\(\s*SELECT', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Performance',
                description='IN with subquery can be slow for large datasets',
                recommendation='Consider using JOIN or EXISTS instead',
                auto_fixable=False,
                estimated_impact='Can reduce execution time by 50%+ for large datasets'
            ))
        
        if re.search(r'\bNOT\s+IN\s*\(\s*SELECT', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='high',
                category='Performance',
                description='NOT IN with subquery can be very slow and may not handle NULLs correctly',
                recommendation='Use NOT EXISTS or LEFT JOIN with NULL check instead',
                auto_fixable=False,
                estimated_impact='Can reduce execution time by 70%+ for large datasets'
            ))
    
    def _check_distinct_usage(self, query: str):
        """Check for DISTINCT usage"""
        if re.search(r'\bSELECT\s+DISTINCT\b', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='low',
                category='Performance',
                description='DISTINCT requires sorting/hashing of results',
                recommendation='Verify if DISTINCT is necessary; consider fixing data model if used frequently',
                auto_fixable=False,
                estimated_impact='Adds overhead for deduplication'
            ))
    
    def _check_join_types(self, query: str):
        """Check for join optimization opportunities"""
        if re.search(r'\bCROSS\s+JOIN\b', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='high',
                category='Performance',
                description='CROSS JOIN creates cartesian product',
                recommendation='Ensure CROSS JOIN is intentional; add join conditions if needed',
                auto_fixable=False,
                estimated_impact='Can cause exponential result set growth'
            ))
        
        # Check for comma joins (old style)
        from_match = re.search(r'\bFROM\s+(\w+)\s*,\s*(\w+)', query, re.IGNORECASE)
        if from_match:
            self.issues.append(OptimizationIssue(
                severity='low',
                category='Best Practice',
                description='Using comma-separated tables (implicit join)',
                recommendation='Use explicit JOIN syntax for better readability',
                auto_fixable=True,
                estimated_impact='Improves maintainability'
            ))
    
    def _check_index_hints(self, query: str):
        """Check for missing index opportunities"""
        where_cols = re.findall(r'\bWHERE\s+(?:\w+\.)?(\w+)', query, re.IGNORECASE)
        join_cols = re.findall(r'\bON\s+\w+\.(\w+)\s*=', query, re.IGNORECASE)
        
        if where_cols or join_cols:
            self.issues.append(OptimizationIssue(
                severity='low',
                category='Information',
                description='Ensure indexes exist on WHERE and JOIN columns',
                recommendation=f'Consider indexes on: {", ".join(set(where_cols + join_cols))}',
                auto_fixable=False,
                estimated_impact='Proper indexes can reduce query time by 90%+'
            ))
    
    def _check_like_patterns(self, query: str):
        """Check for inefficient LIKE patterns"""
        if re.search(r'LIKE\s+["\']%', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='high',
                category='Performance',
                description='LIKE with leading wildcard prevents index usage',
                recommendation='Avoid leading wildcards in LIKE patterns when possible; consider full-text search',
                auto_fixable=False,
                estimated_impact='Forces full table scan'
            ))
    
    def _check_union_vs_union_all(self, query: str):
        """Check for UNION vs UNION ALL"""
        if re.search(r'\bUNION\s+(?!ALL)', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Performance',
                description='UNION removes duplicates (expensive operation)',
                recommendation='Use UNION ALL if duplicates are acceptable',
                auto_fixable=True,
                estimated_impact='UNION ALL can be 2-3x faster'
            ))
    
    def _check_aggregation_issues(self, query: str):
        """Check for aggregation performance issues"""
        if re.search(r'\bGROUP\s+BY\b', query, re.IGNORECASE):
            group_by_match = re.search(r'GROUP\s+BY\s+(.+?)(?:ORDER|HAVING|LIMIT|;|$)', 
                                      query, re.IGNORECASE | re.DOTALL)
            if group_by_match:
                cols = [c.strip() for c in group_by_match.group(1).split(',')]
                if len(cols) > 5:
                    self.issues.append(OptimizationIssue(
                        severity='medium',
                        category='Performance',
                        description=f'GROUP BY with many columns ({len(cols)}) can be slow',
                        recommendation='Consider if all grouping columns are necessary',
                        auto_fixable=False,
                        estimated_impact='Reduces memory usage and sort time'
                    ))
    
    def _check_null_checks(self, query: str):
        """Check for NULL handling"""
        if re.search(r'=\s*NULL|NULL\s*=', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Correctness',
                description='Using = NULL instead of IS NULL',
                recommendation='Use IS NULL or IS NOT NULL for NULL checks',
                auto_fixable=True,
                estimated_impact='Fixes logical errors - critical for correctness'
            ))
    
    def _check_nested_loops(self, query: str):
        """Check for potential nested loop issues"""
        subquery_count = len(re.findall(r'\(\s*SELECT', query, re.IGNORECASE))
        if subquery_count >= 3:
            self.issues.append(OptimizationIssue(
                severity='high',
                category='Performance',
                description=f'Multiple nested subqueries ({subquery_count}) detected',
                recommendation='Consider using CTEs (WITH clause) for better readability and potential optimization',
                auto_fixable=False,
                estimated_impact='CTEs can be cached by optimizer'
            ))
    
    def _check_redundant_conditions(self, query: str):
        """Check for redundant WHERE conditions"""
        # Look for duplicate conditions
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|;|$)', query, re.IGNORECASE | re.DOTALL)
        if where_match:
            conditions = where_match.group(1)
            # Simple check for obvious duplicates
            if re.search(r'(\w+\.\w+\s*=\s*\S+).*\1', conditions, re.IGNORECASE):
                self.issues.append(OptimizationIssue(
                    severity='low',
                    category='Code Quality',
                    description='Potential redundant WHERE conditions detected',
                    recommendation='Remove duplicate conditions',
                    auto_fixable=False,
                    estimated_impact='Improves maintainability'
                ))
    
    def _check_database_specific(self, query: str):
        """Check for database-specific optimizations"""
        if self.db_type == DatabaseType.POSTGRESQL:
            if re.search(r'\|\|', query):
                self.issues.append(OptimizationIssue(
                    severity='low',
                    category='Best Practice',
                    description='String concatenation in query',
                    recommendation='Consider using CONCAT() for better portability',
                    auto_fixable=False
                ))
            
            # Check for missing VACUUM opportunities
            if re.search(r'\b(UPDATE|DELETE)\b', query, re.IGNORECASE):
                self.issues.append(OptimizationIssue(
                    severity='low',
                    category='Maintenance',
                    description='Heavy UPDATE/DELETE operations detected',
                    recommendation='Consider running VACUUM ANALYZE periodically',
                    auto_fixable=False
                ))
        
        elif self.db_type == DatabaseType.MYSQL:
            if re.search(r'\b(DELETE|UPDATE)\b', query, re.IGNORECASE):
                if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
                    self.issues.append(OptimizationIssue(
                        severity='low',
                        category='Safety',
                        description='DELETE/UPDATE without LIMIT',
                        recommendation='Consider adding LIMIT clause for safety in MySQL',
                        auto_fixable=False
                    ))
        
        elif self.db_type == DatabaseType.SNOWFLAKE:
            if re.search(r'\bWHERE\s+\w+\s+BETWEEN', query, re.IGNORECASE):
                self.issues.append(OptimizationIssue(
                    severity='low',
                    category='Information',
                    description='Range query detected',
                    recommendation='Consider clustering key on range-queried columns',
                    auto_fixable=False
                ))
            
            # Check for result set size
            if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
                self.issues.append(OptimizationIssue(
                    severity='low',
                    category='Cost',
                    description='No LIMIT clause in Snowflake query',
                    recommendation='Consider adding LIMIT to control costs and result set size',
                    auto_fixable=False
                ))
    
    def _check_data_lake_optimizations(self, query: str):
        """Detect Data Lake patterns and provide optimization recommendations.

        Focuses on Snowflake and modern lakehouse usage patterns like CTAS, external stages,
        partition pruning, file formats, COPY INTO, and clustering. Also provides general
        lake optimization advice when patterns suggest large-scale data scans.
        """
        q = query.upper()

        # CTAS heavy operations: advise clustering and column pruning
        if re.search(r"CREATE\s+TABLE\s+.*\s+AS\s+SELECT", q) or re.search(r"\bCTAS\b", q):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Data Lake',
                description='CTAS detected; consider clustering and column pruning',
                recommendation='Use CLUSTER BY on selective columns; select only required columns',
                query_section='CTAS',
                auto_fixable=False,
                estimated_impact='10-50% scan reduction with effective clustering'
            ))

        # External stage usage and ingestion optimization
        if re.search(r"\bCOPY\s+INTO\b", q) or re.search(r"\bFROM\s+@", q) or re.search(r"S3://|GCS://|AZURE://", q):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Data Lake',
                description='External stage or lake source detected; optimize ingestion and scans',
                recommendation='Prefer PARQUET/ORC; apply partition filters in WHERE; avoid SELECT *',
                query_section='External stage',
                auto_fixable=False,
                estimated_impact='30-80% cost reduction via pruning and columnar formats'
            ))

        # Columnar file formats: advise column pruning
        if re.search(r"\b(PARQUET|ORC|DELTA)\b", q):
            self.issues.append(OptimizationIssue(
                severity='low',
                category='Data Lake',
                description='Columnar file format detected; ensure column pruning is used',
                recommendation='Explicitly select needed columns; avoid SELECT * on columnar data',
                query_section='File format',
                auto_fixable=False
            ))

        # Partition pruning patterns: date/time filters
        if re.search(r"\bWHERE\s+.*(DATE|TIMESTAMP|EVENT_DATE|PARTITION)\s*(=|BETWEEN|>=|<=)", q):
            self.issues.append(OptimizationIssue(
                severity='low',
                category='Data Lake',
                description='Partition filter detected; verify alignment with clustering/partitioning',
                recommendation='Ensure clustered/partitioned columns match common query predicates',
                query_section='Partition pruning',
                auto_fixable=False
            ))

        # Missing LIMIT on exploratory wide aggregations
        if re.search(r"\bSELECT\b\s+.*\bFROM\b\s+", q) and not re.search(r"\bLIMIT\b\s+\d+", q):
            if re.search(r"\bGROUP\s+BY\b|\bOVER\s*\(", q):
                self.issues.append(OptimizationIssue(
                    severity='low',
                    category='Data Lake',
                    description='Wide aggregations without LIMIT can scan large datasets',
                    recommendation='Use LIMIT for exploration and pre-filter with selective WHERE clauses',
                    query_section='Aggregation without LIMIT',
                    auto_fixable=False
                ))

        # Snowflake-specific clustering and caching hints
        if self.db_type == DatabaseType.SNOWFLAKE:
            if re.search(r"\bFROM\b\s+\w+", q) and not re.search(r"\bCLUSTER\s+BY\b", q):
                self.issues.append(OptimizationIssue(
                    severity='low',
                    category='Snowflake',
                    description='Consider CLUSTER BY on large tables for better pruning',
                    recommendation='Identify high-selectivity columns and apply CLUSTER BY for large fact tables',
                    query_section='Clustering',
                    auto_fixable=False
                ))

            # Result caching advisory
            if re.search(r"\bSELECT\b", q):
                self.issues.append(OptimizationIssue(
                    severity='info',
                    category='Snowflake',
                    description='Result caching can accelerate repeated queries',
                    recommendation='Leverage result caching when data freshness permits; avoid unnecessary session changes',
                    query_section='Caching',
                    auto_fixable=False
                ))

    def generate_report(self, issues: List[OptimizationIssue], show_impact: bool = True) -> str:
        """Generate a formatted report of issues"""
        if not issues:
            return "✓ No optimization issues found!"
        
        report = ["SQL Optimization Report", "=" * 70, ""]
        
        # Group by severity
        by_severity = {'critical': [], 'high': [], 'medium': [], 'low': []}
        for issue in issues:
            by_severity[issue.severity].append(issue)
        
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_issues = by_severity[severity]
            if not severity_issues:
                continue
            
            report.append(f"\n{severity.upper()} ({len(severity_issues)} issues)")
            report.append("-" * 70)
            
            for i, issue in enumerate(severity_issues, 1):
                report.append(f"\n{i}. [{issue.category}] {issue.description}")
                report.append(f"   → {issue.recommendation}")
                
                if issue.query_section:
                    report.append(f"   Section: {issue.query_section}")
                
                if show_impact and issue.estimated_impact:
                    report.append(f"   Impact: {issue.estimated_impact}")
                
                if issue.auto_fixable:
                    report.append(f"   ✓ Auto-fixable")
        
        report.append(f"\n\nTotal issues found: {len(issues)}")
        auto_fixable = sum(1 for i in issues if i.auto_fixable)
        if auto_fixable:
            report.append(f"Auto-fixable issues: {auto_fixable}")
        
        return "\n".join(report)


class DatabaseConnector:
    """Connect to databases and analyze queries in production"""
    
    def __init__(self, db_type: DatabaseType, connection_params: Dict[str, Any]):
        self.db_type = db_type
        self.connection_params = connection_params
        self.optimizer = SQLOptimizer(db_type)
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            if self.db_type == DatabaseType.POSTGRESQL:
                import psycopg2
                self.connection = psycopg2.connect(**self.connection_params)
            
            elif self.db_type == DatabaseType.MYSQL:
                import pymysql
                self.connection = pymysql.connect(**self.connection_params)
            
            elif self.db_type == DatabaseType.SQLSERVER:
                import pyodbc
                conn_str = ';'.join([f"{k}={v}" for k, v in self.connection_params.items()])
                self.connection = pyodbc.connect(conn_str)
            
            elif self.db_type == DatabaseType.SNOWFLAKE:
                import snowflake.connector
                self.connection = snowflake.connector.connect(**self.connection_params)
            
            print(f"✓ Connected to {self.db_type.value}")
            return True
        
        except Exception as e:
            print(f"✗ Connection failed: {str(e)}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def get_execution_plan(self, query: str) -> Dict:
        """Get query execution plan"""
        if not self.connection:
            raise Exception("Not connected to database")
        
        cursor = self.connection.cursor()
        plan = {}
        
        try:
            if self.db_type == DatabaseType.POSTGRESQL:
                cursor.execute(f"EXPLAIN (FORMAT JSON, ANALYZE) {query}")
                plan = cursor.fetchone()[0]
            
            elif self.db_type == DatabaseType.MYSQL:
                cursor.execute(f"EXPLAIN FORMAT=JSON {query}")
                plan = cursor.fetchone()[0]
            
            elif self.db_type == DatabaseType.SQLSERVER:
                cursor.execute("SET STATISTICS PROFILE ON")
                cursor.execute(query)
                plan = cursor.fetchall()
            
            elif self.db_type == DatabaseType.SNOWFLAKE:
                cursor.execute(f"EXPLAIN {query}")
                plan = {'plan': cursor.fetchall()}
        
        except Exception as e:
            plan = {'error': str(e)}
        
        finally:
            cursor.close()
        
        return plan
    
    def analyze_slow_queries(self, threshold_ms: int = 1000, limit: int = 10) -> List[Dict]:
        """Analyze slow queries from database logs"""
        if not self.connection:
            raise Exception("Not connected to database")
        
        cursor = self.connection.cursor()
        slow_queries = []
        
        try:
            if self.db_type == DatabaseType.POSTGRESQL:
                # Query pg_stat_statements
                query = """
                    SELECT 
                        query,
                        calls,
                        total_exec_time / 1000 as total_time_sec,
                        mean_exec_time as avg_time_ms,
                        max_exec_time as max_time_ms,
                        rows
                    FROM pg_stat_statements
                    WHERE mean_exec_time > %s
                    ORDER BY mean_exec_time DESC
                    LIMIT %s
                """
                cursor.execute(query, (threshold_ms, limit))
                
                for row in cursor.fetchall():
                    slow_queries.append({
                        'query': row[0],
                        'calls': row[1],
                        'total_time_sec': float(row[2]),
                        'avg_time_ms': float(row[3]),
                        'max_time_ms': float(row[4]),
                        'rows': row[5]
                    })
            
            elif self.db_type == DatabaseType.MYSQL:
                # Query slow query log table
                query = """
                    SELECT 
                        sql_text,
                        query_time,
                        lock_time,
                        rows_sent,
                        rows_examined
                    FROM mysql.slow_log
                    WHERE query_time > %s
                    ORDER BY query_time DESC
                    LIMIT %s
                """
                cursor.execute(query, (threshold_ms / 1000.0, limit))
                
                for row in cursor.fetchall():
                    slow_queries.append({
                        'query': row[0],
                        'query_time_sec': float(row[1].total_seconds()),
                        'lock_time_sec': float(row[2].total_seconds()),
                        'rows_sent': row[3],
                        'rows_examined': row[4]
                    })
            
            elif self.db_type == DatabaseType.SQLSERVER:
                # Query DMVs
                query = """
                    SELECT TOP %s
                        qt.text AS query,
                        qs.execution_count,
                        qs.total_elapsed_time / 1000000.0 as total_time_sec,
                        qs.total_elapsed_time / qs.execution_count / 1000.0 as avg_time_ms,
                        qs.total_logical_reads,
                        qs.total_physical_reads
                    FROM sys.dm_exec_query_stats qs
                    CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) qt
                    WHERE qs.total_elapsed_time / qs.execution_count / 1000.0 > %s
                    ORDER BY qs.total_elapsed_time / qs.execution_count DESC
                """
                cursor.execute(query, (limit, threshold_ms))
                
                for row in cursor.fetchall():
                    slow_queries.append({
                        'query': row[0],
                        'execution_count': row[1],
                        'total_time_sec': float(row[2]),
                        'avg_time_ms': float(row[3]),
                        'logical_reads': row[4],
                        'physical_reads': row[5]
                    })
            
            elif self.db_type == DatabaseType.SNOWFLAKE:
                # Query QUERY_HISTORY
                query = f"""
                    SELECT 
                        query_text,
                        execution_time / 1000.0 as execution_time_sec,
                        compilation_time / 1000.0 as compilation_time_sec,
                        bytes_scanned,
                        rows_produced
                    FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
                    WHERE execution_time > {threshold_ms}
                    ORDER BY execution_time DESC
                    LIMIT {limit}
                """
                cursor.execute(query)
                
                for row in cursor.fetchall():
                    slow_queries.append({
                        'query': row[0],
                        'execution_time_sec': float(row[1]),
                        'compilation_time_sec': float(row[2]),
                        'bytes_scanned': row[3],
                        'rows_produced': row[4]
                    })
        
        except Exception as e:
            print(f"Error querying slow queries: {str(e)}")
        
        finally:
            cursor.close()
        
        return slow_queries
    
    def get_table_statistics(self, table_name: str) -> Dict:
        """Get table statistics for optimization insights"""
        if not self.connection:
            raise Exception("Not connected to database")
        
        cursor = self.connection.cursor()
        stats = {}
        
        try:
            if self.db_type == DatabaseType.POSTGRESQL:
                # Get table size and statistics
                query = f"""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('{table_name}')) as total_size,
                        n_live_tup as row_count,
                        n_dead_tup as dead_rows,
                        last_vacuum,
                        last_analyze
                    FROM pg_stat_user_tables
                    WHERE relname = '{table_name}'
                """
                cursor.execute(query)
                row = cursor.fetchone()
                
                if row:
                    stats = {
                        'total_size': row[0],
                        'row_count': row[1],
                        'dead_rows': row[2],
                        'last_vacuum': row[3],
                        'last_analyze': row[4]
                    }
                
                # Get index information
                cursor.execute(f"""
                    SELECT indexname, indexdef 
                    FROM pg_indexes 
                    WHERE tablename = '{table_name}'
                """)
                stats['indexes'] = [{'name': r[0], 'definition': r[1]} for r in cursor.fetchall()]
            
            elif self.db_type == DatabaseType.MYSQL:
                cursor.execute(f"SHOW TABLE STATUS LIKE '{table_name}'")
                row = cursor.fetchone()
                
                if row:
                    stats = {
                        'engine': row[1],
                        'row_count': row[4],
                        'data_length': row[6],
                        'index_length': row[8],
                        'data_free': row[9]
                    }
                
                cursor.execute(f"SHOW INDEX FROM {table_name}")
                stats['indexes'] = [{'name': r[2], 'column': r[4]} for r in cursor.fetchall()]
            
            elif self.db_type == DatabaseType.SNOWFLAKE:
                cursor.execute(f"""
                    SELECT 
                        ROW_COUNT,
                        BYTES,
                        CLUSTERING_KEY
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_NAME = '{table_name.upper()}'
                """)
                row = cursor.fetchone()
                
                if row:
                    stats = {
                        'row_count': row[0],
                        'bytes': row[1],
                        'clustering_key': row[2]
                    }
        
        except Exception as e:
            stats['error'] = str(e)
        
        finally:
            cursor.close()
        
        return stats
    
    def monitor_performance_realtime(self, duration_sec: int = 60):
        """Monitor database performance in real-time"""
        import time
        
        if not self.connection:
            raise Exception("Not connected to database")
        
        print(f"Monitoring performance for {duration_sec} seconds...")
        start_time = time.time()
        metrics = []
        
        while time.time() - start_time < duration_sec:
            cursor = self.connection.cursor()
            
            try:
                if self.db_type == DatabaseType.POSTGRESQL:
                    cursor.execute("""
                        SELECT 
                            count(*) as active_connections,
                            sum(case when state = 'active' then 1 else 0 end) as active_queries,
                            sum(case when wait_event_type IS NOT NULL then 1 else 0 end) as waiting_queries
                        FROM pg_stat_activity
                        WHERE datname = current_database()
                    """)
                    row = cursor.fetchone()
                    
                    metrics.append({
                        'timestamp': datetime.now(),
                        'active_connections': row[0],
                        'active_queries': row[1],
                        'waiting_queries': row[2]
                    })
                
                elif self.db_type == DatabaseType.MYSQL:
                    cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
                    connections = cursor.fetchone()[1]
                    
                    cursor.execute("SHOW STATUS LIKE 'Threads_running'")
                    running = cursor.fetchone()[1]
                    
                    metrics.append({
                        'timestamp': datetime.now(),
                        'connections': int(connections),
                        'running_threads': int(running)
                    })
            
            except Exception as e:
                print(f"Monitoring error: {str(e)}")
            
            finally:
                cursor.close()
            
            time.sleep(5)  # Sample every 5 seconds
        
        return metrics


class QueryOptimizationSuite:
    """Complete suite for SQL optimization"""
    
    def __init__(self, db_type: DatabaseType, connection_params: Optional[Dict] = None):
        self.optimizer = SQLOptimizer(db_type)
        self.connector = DatabaseConnector(db_type, connection_params) if connection_params else None
        self.regression_tester = RegressionTester(db_type)
        self.workload_analyzer = WorkloadAnalyzer()
        self.index_creator = AutomatedIndexCreator(db_type)
        self.query_history: List[QueryAnalysis] = []
    
    def analyze_and_optimize(self, query: str, auto_fix: bool = True, 
                            get_execution_plan: bool = False,
                            table_sizes: Optional[Dict[str, int]] = None) -> Dict:
        """Complete analysis and optimization workflow"""
        
        # Comprehensive analysis
        analysis = self.optimizer.analyze_query_comprehensive(query, auto_fix=auto_fix)
        self.query_history.append(analysis)
        
        # Get execution plan if connected
        execution_time = None
        if get_execution_plan and self.connector and self.connector.connection:
            analysis.execution_plan = self.connector.get_execution_plan(query)
        
        # Get index recommendations
        index_recommendations = self.optimizer.index_recommender.analyze_query_for_indexes(query)
        
        # Caching analysis
        caching_analysis = self.optimizer.caching_analyzer.analyze_caching_potential(
            query, execution_time
        )
        
        # Parallel execution analysis
        parallel_analysis = self.optimizer.parallel_analyzer.analyze_parallelization(
            query, analysis.execution_plan
        )
        
        # Cost estimation
        cost_analysis = self.optimizer.cost_estimator.estimate_cost(query, table_sizes)
        
        # Build comprehensive result
        result = {
            'original_query': analysis.original_query,
            'query_hash': analysis.query_hash,
            'complexity_score': analysis.complexity_score,
            'detected_patterns': [p.value for p in analysis.detected_patterns],
            'issues_found': len(analysis.issues),
            'issues': analysis.issues,
            'optimized_query': analysis.optimized_query,
            'index_recommendations': index_recommendations,
            'execution_plan': analysis.execution_plan,
            'caching_recommendations': caching_analysis,
            'parallel_execution': parallel_analysis,
            'cost_estimate': cost_analysis
        }
        
        return result
    
    def analyze_workload_from_logs(self, log_queries: List[Tuple[str, float]], 
                                   time_period_hours: int = 24) -> Dict:
        """Analyze workload from query logs"""
        
        # Add queries to workload analyzer
        for query, exec_time in log_queries:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            patterns = self.optimizer.pattern_detector.detect_patterns(query, query_hash)
            self.workload_analyzer.add_query_execution(query, exec_time, patterns)
        
        # Analyze workload
        workload_analysis = self.workload_analyzer.analyze_workload(time_period_hours)
        
        # Identify optimization opportunities
        opportunities = self.workload_analyzer.identify_optimization_opportunities()
        
        # Detect N+1 queries
        all_queries = [q for q, _ in log_queries]
        n_plus_1_analysis = self.optimizer.pattern_detector.detect_n_plus_1(all_queries)
        
        return {
            'workload_analysis': workload_analysis,
            'optimization_opportunities': opportunities,
            'n_plus_1_detection': n_plus_1_analysis
        }
    
    def generate_index_recommendations_from_workload(self, queries: List[str],
                                                    frequencies: Optional[Dict[str, int]] = None) -> List[IndexRecommendation]:
        """Generate index recommendations based on workload"""
        
        return self.index_creator.analyze_workload_for_indexes(queries, frequencies)
    
    def auto_create_indexes(self, recommendations: Optional[List[IndexRecommendation]] = None,
                           dry_run: bool = True, max_indexes: int = 10) -> Dict:
        """Automatically create recommended indexes"""
        
        if not self.connector or not self.connector.connection:
            return {'error': 'Database connection required'}
        
        return self.index_creator.create_indexes(
            self.connector.connection,
            recommendations,
            dry_run=dry_run,
            max_indexes=max_indexes
        )
    
    def test_optimization(self, original_query: str, optimized_query: str,
                         iterations: int = 5) -> Dict:
        """Run regression test on optimized query"""
        
        if not self.connector or not self.connector.connection:
            return {'error': 'Database connection required for regression testing'}
        
        return self.regression_tester.run_regression_test(
            self.connector.connection,
            original_query,
            optimized_query,
            iterations
        )
    
    def full_optimization_workflow(self, query: str, test_optimization: bool = True) -> Dict:
        """Complete workflow: analyze, optimize, and test"""
        
        workflow_result = {
            'step1_analysis': None,
            'step2_optimization': None,
            'step3_testing': None,
            'recommendations': []
        }
        
        # Step 1: Analyze
        print("Step 1: Analyzing query...")
        analysis = self.analyze_and_optimize(query, auto_fix=True, get_execution_plan=True)
        workflow_result['step1_analysis'] = analysis
        
        # Step 2: Generate recommendations
        print("Step 2: Generating recommendations...")
        recommendations = []
        
        if analysis['issues_found'] > 0:
            recommendations.append(f"Found {analysis['issues_found']} optimization opportunities")
        
        if analysis['caching_recommendations']['cacheable']:
            recommendations.append(
                f"Query is cacheable: {analysis['caching_recommendations']['cache_strategy']}"
            )
        
        if analysis['parallel_execution']['parallelizable']:
            recommendations.append(
                f"Query can benefit from parallel execution"
            )
        
        if analysis['cost_estimate']['cost_level'] in ['high', 'very_high']:
            recommendations.append(
                f"High cost query ({analysis['cost_estimate']['cost_level']}) - prioritize optimization"
            )
        
        workflow_result['recommendations'] = recommendations
        
        # Step 3: Test if optimized query was generated
        if test_optimization and analysis['optimized_query'] and self.connector:
            print("Step 3: Running regression tests...")
            test_result = self.test_optimization(
                query,
                analysis['optimized_query']
            )
            workflow_result['step3_testing'] = test_result
        
        return workflow_result
    
    def batch_analyze_file(self, filepath: str, auto_fix: bool = True,
                           output_file: Optional[str] = None,
                           get_execution_plan: bool = False) -> List[Dict]:
        """Analyze all queries in a `.sql` file and optionally save results.
        
        - Splits the file into statements, runs comprehensive analysis per query,
          and augments results with index, caching, parallelization, and cost.
        - If `get_execution_plan` is True and a live connection exists, includes plans.
        - Saves JSON output to `output_file` if provided.
        """
        # Read and split queries using the v3 splitter to respect SQL semantics
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        queries = self.optimizer._split_queries(content)

        results: List[Dict] = []
        for q in queries:
            if not q.strip():
                continue

            # Core analysis with auto-fix
            analysis = self.optimizer.analyze_query_comprehensive(q, auto_fix=auto_fix)

            # Optionally retrieve execution plan if connected
            plan: Optional[Dict] = None
            if get_execution_plan and self.connector and self.connector.connection:
                try:
                    plan = self.connector.get_execution_plan(q)
                    analysis.execution_plan = plan
                except Exception as e:
                    plan = {'error': str(e)}

            # Enrich with recommendations and estimates
            index_recs = self.optimizer.index_recommender.analyze_query_for_indexes(q, plan)
            caching = self.optimizer.caching_analyzer.analyze_caching_potential(q)
            parallel = self.optimizer.parallel_analyzer.analyze_parallelization(q, plan)
            cost = self.optimizer.cost_estimator.estimate_cost(q)

            # Assemble result compatible with demo expectations
            results.append({
                'original_query': analysis.original_query,
                'query_hash': analysis.query_hash,
                'complexity_score': analysis.complexity_score,
                'detected_patterns': [p.value for p in analysis.detected_patterns],
                'issues_found': len(analysis.issues),
                'issues': [
                    {
                        'severity': i.severity,
                        'category': i.category,
                        'description': i.description,
                        'recommendation': i.recommendation,
                        'auto_fixable': i.auto_fixable,
                    }
                    for i in analysis.issues
                ],
                'optimized_query': analysis.optimized_query,
                'index_recommendations': index_recs,
                'execution_plan': plan,
                'caching_recommendations': caching,
                'parallel_execution': parallel,
                'cost_estimate': cost,
            })

        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {output_file}")

        return results
    
    def generate_comprehensive_report(self, analyses: List[QueryAnalysis]) -> str:
        """Generate detailed report for multiple queries"""
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE SQL OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total queries analyzed: {len(analyses)}")
        report.append("")
        
        # Summary statistics
        total_issues = sum(len(a.issues) for a in analyses)
        critical_issues = sum(len([i for i in a.issues if i.severity == 'critical']) for a in analyses)
        auto_fixable = sum(len([i for i in a.issues if i.auto_fixable]) for a in analyses)
        avg_complexity = sum(a.complexity_score for a in analyses) / len(analyses) if analyses else 0
        
        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Total issues found: {total_issues}")
        report.append(f"Critical issues: {critical_issues}")
        report.append(f"Auto-fixable issues: {auto_fixable}")
        report.append(f"Average complexity score: {avg_complexity:.1f}/100")
        report.append("")
        
        # Individual query reports
        for i, analysis in enumerate(analyses, 1):
            report.append(f"\nQUERY #{i}")
            report.append("-" * 80)
            report.append(f"Hash: {analysis.query_hash}")
            report.append(f"Complexity: {analysis.complexity_score}/100")
            report.append(f"Issues: {len(analysis.issues)}")
            report.append("")
            
            if analysis.issues:
                report.append(self.optimizer.generate_report(analysis.issues))
            else:
                report.append("✓ No issues found")
            
            if analysis.optimized_query:
                report.append("\nOPTIMIZED VERSION:")
                report.append(analysis.optimized_query)
            
            report.append("")
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    print("SQL Optimization Tool - ML/Backend Engineer Edition")
    print("=" * 80)
    
    # Example 1: Query Pattern Detection
    print("\n1. QUERY PATTERN DETECTION")
    print("-" * 80)
    
    suite = QueryOptimizationSuite(DatabaseType.POSTGRESQL)
    
    sample_queries = [
        ("Analytics Query", """
            SELECT 
                user_id,
                COUNT(*) as total_orders,
                SUM(amount) as total_spent,
                AVG(amount) as avg_order
            FROM orders
            WHERE order_date >= '2024-01-01'
            GROUP BY user_id
            HAVING COUNT(*) > 5
        """),
        ("N+1 Pattern", "SELECT * FROM users WHERE id = 1"),
        ("Complex Join", """
            SELECT u.*, o.*, p.*
            FROM users u
            JOIN orders o ON u.id = o.user_id
            JOIN products p ON o.product_id = p.id
            JOIN categories c ON p.category_id = c.id
            WHERE u.status = 'active'
        """),
        ("Time Series", """
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour,
                COUNT(*) as events
            FROM events
            WHERE timestamp BETWEEN '2024-01-01' AND '2024-01-31'
            GROUP BY hour
        """)
    ]
    
    for name, query in sample_queries:
        result = suite.analyze_and_optimize(query)
        print(f"\n{name}:")
        print(f"  Patterns: {', '.join(result['detected_patterns'])}")
        print(f"  Complexity: {result['complexity_score']}/100")
        print(f"  Issues: {result['issues_found']}")
    
    # Example 2: Workload Analysis
    print("\n\n2. WORKLOAD ANALYSIS")
    print("-" * 80)
    
    # Simulate query logs (query, execution_time_ms)
    query_logs = [
        ("SELECT * FROM users WHERE id = ?", 5.2),
        ("SELECT * FROM orders WHERE user_id = ?", 150.0),
        ("SELECT * FROM orders WHERE user_id = ?", 145.0),
        ("SELECT * FROM orders WHERE user_id = ?", 155.0),
        ("SELECT COUNT(*) FROM orders", 2000.0),
        ("SELECT * FROM products WHERE category_id = ?", 80.0),
    ] * 20  # Simulate 120 queries
    
    workload_result = suite.analyze_workload_from_logs(query_logs, time_period_hours=1)
    
    print("\nWorkload Summary:")
    summary = workload_result['workload_analysis']['summary']
    print(f"  Total queries: {summary['total_queries']}")
    print(f"  Avg time: {summary['avg_time_ms']:.2f}ms")
    print(f"  Queries/hour: {summary.get('queries_per_hour', 'N/A')}")
    
    print("\nPercentiles:")
    perc = workload_result['workload_analysis']['percentiles']
    print(f"  P50: {perc['p50_ms']:.2f}ms")
    print(f"  P95: {perc['p95_ms']:.2f}ms")
    print(f"  P99: {perc['p99_ms']:.2f}ms")
    
    print("\nTop Optimization Opportunities:")
    for i, opp in enumerate(workload_result['optimization_opportunities'][:3], 1):
        print(f"  {i}. {opp['type']}")
        print(f"     Frequency: {opp.get('frequency', 'N/A')}")
        print(f"     Avg time: {opp.get('avg_time_ms', 'N/A')}")
        print(f"     {opp['recommendation']}")
    
    # Check for N+1 queries
    if workload_result['n_plus_1_detection']['detected']:
        print("\n⚠️  N+1 Query Problem Detected!")
        for candidate in workload_result['n_plus_1_detection']['candidates']:
            print(f"  Pattern repeated {candidate['occurrences']} times")
            print(f"  Severity: {candidate['severity']}")
    
    # Example 3: Automated Index Creation
    print("\n\n3. AUTOMATED INDEX CREATION")
    print("-" * 80)
    
    # Sample queries for index analysis
    workload_queries = [
        "SELECT * FROM users WHERE email = 'test@example.com'",
        "SELECT * FROM orders WHERE user_id = 123 AND status = 'completed'",
        "SELECT * FROM orders WHERE created_at > '2024-01-01'",
        "SELECT * FROM users u JOIN orders o ON u.id = o.user_id WHERE u.status = 'active'",
        "SELECT * FROM products WHERE category_id = 5 ORDER BY price DESC",
    ]
    
    # Generate recommendations
    index_recs = suite.generate_index_recommendations_from_workload(workload_queries)
    
    print(f"\nGenerated {len(index_recs)} index recommendations:")
    for i, rec in enumerate(index_recs[:5], 1):
        print(f"\n  {i}. [{rec.priority.upper()}] {rec.table}.{', '.join(rec.columns)}")
        print(f"     Type: {rec.index_type}")
        print(f"     Reason: {rec.reason}")
        print(f"     Benefit: {rec.estimated_benefit}")
        print(f"     SQL: {rec.sql_create}")
    
    print("\n\n4. INDEX CREATION (DRY RUN)")
    print("-" * 80)
    print("""
# To actually create indexes with database connection:

suite_with_db = QueryOptimizationSuite(
    DatabaseType.POSTGRESQL,
    {'host': 'localhost', 'database': 'mydb', 
     'user': 'postgres', 'password': 'password'}
)

suite_with_db.connector.connect()

# Create top 5 indexes (dry run)
result = suite_with_db.auto_create_indexes(dry_run=True, max_indexes=5)
print(f"Would create {len(result['dry_run_sql'])} indexes")

# Actually create indexes (removes dry_run)
result = suite_with_db.auto_create_indexes(dry_run=False, max_indexes=5)
print(f"Created: {len(result['created'])} indexes")
print(f"Failed: {len(result['failed'])} indexes")
print(f"Skipped: {len(result['skipped'])} indexes")

# Drop unused indexes
drop_result = suite_with_db.index_creator.drop_unused_indexes(
    suite_with_db.connector.connection,
    min_scans=100,
    dry_run=True
)
    """)
    
    # Example 5: Pattern-Specific Optimizations
    print("\n5. PATTERN-SPECIFIC RECOMMENDATIONS")
    print("-" * 80)
    
    pattern_recommendations = {
        'analytics_aggregation': [
            "Consider materialized views for frequently accessed aggregations",
            "Use columnar storage (e.g., PostgreSQL with cstore_fdw)",
            "Partition large tables by date for time-based aggregations"
        ],
        'n_plus_1': [
            "Use eager loading with JOINs instead of separate queries",
            "Implement DataLoader pattern for batching (GraphQL)",
            "Use SELECT ... WHERE id IN (?,?,?) for batch loading"
        ],
        'full_table_scan': [
            "Add WHERE clause to filter data",
            "Create appropriate indexes on filter columns",
            "Consider table partitioning for very large tables"
        ],
        'search_pattern': [
            "Use full-text search indexes (GIN/GiST in PostgreSQL)",
            "Consider Elasticsearch for complex search requirements",
            "Avoid leading wildcards in LIKE queries"
        ]
    }
    
    print("\nRecommendations by Pattern:")
    for pattern, recs in pattern_recommendations.items():
        print(f"\n  {pattern.upper()}:")
        for rec in recs:
            print(f"    • {rec}")
    
    # Example 6: ML/Backend Engineer Workflow
    print("\n\n6. TYPICAL ML/BACKEND ENGINEER WORKFLOW")
    print("-" * 80)
    print("""
# Scenario: You're building a feature extraction pipeline for ML

# Step 1: Analyze your data extraction query
query = '''
    SELECT 
        user_id,
        COUNT(*) as feature_1,
        AVG(amount) as feature_2,
        MAX(timestamp) as feature_3
    FROM events
    WHERE event_type = 'purchase'
    GROUP BY user_id
'''

result = suite.analyze_and_optimize(query, auto_fix=True)

# Step 2: Check patterns and optimize
print(f"Patterns: {result['detected_patterns']}")
print(f"Cost level: {result['cost_estimate']['cost_level']}")

# If it's a slow analytics query:
if 'analytics_aggregation' in result['detected_patterns']:
    # Consider materialized view
    materialized_view = f'''
    CREATE MATERIALIZED VIEW user_features AS
    {query}
    '''
    
    # Or cache the results
    if result['caching_recommendations']['cacheable']:
        ttl = result['caching_recommendations']['ttl_suggestion']
        # Implement Redis caching with appropriate TTL

# Step 3: Monitor in production
# Collect real query logs
logs = get_production_query_logs(hours=24)
workload = suite.analyze_workload_from_logs(logs)

# Step 4: Auto-optimize
opportunities = workload['optimization_opportunities']
for opp in opportunities[:5]:  # Top 5
    if opp['impact_score'] > 50000:
        # High impact - create index immediately
        pass

# Step 5: Regression test before deploying
test_result = suite.test_optimization(
    original_query=query,
    optimized_query=result['optimized_query']
)

if test_result['passed'] and test_result['comparison']['performance_improvement'] > 20:
    deploy_optimization()
    """)
    
    # Example 7: Real-time Monitoring Dashboard Data
    print("\n7. DATA FOR MONITORING DASHBOARDS")
    print("-" * 80)
    print("""
# Generate metrics for your monitoring dashboard:

dashboard_data = {
    'current_status': {
        'slow_queries': len([q for q in queries if q['time'] > 1000]),
        'avg_query_time': workload_analysis['summary']['avg_time_ms'],
        'p95_latency': workload_analysis['percentiles']['p95_ms'],
        'queries_per_second': total_queries / 3600,
    },
    'alerts': [
        {
            'type': 'n_plus_1_detected',
            'severity': 'high',
            'occurrences': n_plus_1_data['candidates'][0]['occurrences']
        }
    ],
    'index_recommendations': [
        {
            'table': rec.table,
            'columns': rec.columns,
            'priority': rec.priority,
            'estimated_benefit': rec.estimated_benefit
        }
        for rec in index_recommendations[:10]
    ],
    'pattern_distribution': workload_analysis['pattern_distribution']
}

# Send to Grafana, Datadog, or your monitoring system
send_to_monitoring(dashboard_data)
    """)
    
    print("\n" + "=" * 80)
    print("✨ COMPLETE FEATURE LIST FOR ML/BACKEND ENGINEERS:")
    print("=" * 80)
    print("""
✓ Query Optimization & Auto-Fix
✓ Live Database Connections (PostgreSQL, MySQL, SQL Server, Snowflake)
✓ Execution Plan Analysis
✓ Index Recommendations with SQL generation
✓ Query Complexity Scoring
✓ Caching Strategy Recommendations
✓ Parallel Execution Analysis
✓ Cost Estimation (including Snowflake credits)
✓ Automated Regression Testing
✓ Performance Monitoring
✓ Batch File Analysis
✓ Comprehensive Reporting

NEW FEATURES FOR ML/BACKEND ENGINEERS:
✓ Query Pattern Detection (15+ patterns)
✓ N+1 Query Detection
✓ Workload Analysis & Profiling
✓ Optimization Opportunity Identification
✓ Automated Index Creation
✓ Unused Index Detection & Cleanup
✓ Pattern-Specific Recommendations
✓ Impact Scoring & Prioritization
✓ Time-Series Analysis
✓ Peak Load Detection
✓ Query Frequency Analysis
    """)
    
    print("\n📦 Installation:")
    print("pip install sqlparse psycopg2-binary pymysql pyodbc snowflake-connector-python")
    print("\n🎯 Perfect for:")
    print("  • ML Engineers optimizing feature extraction queries")
    print("  • Backend Engineers scaling production databases")
    print("  • DevOps automating database maintenance")
    print("  • Data Engineers optimizing ETL pipelines")
    print("\n" + "=" * 80)
    
    # Example 1: Full analysis with all new features
    print("\n1. COMPREHENSIVE QUERY ANALYSIS")
    print("-" * 80)
    
    suite = QueryOptimizationSuite(DatabaseType.POSTGRESQL)
    
    problematic_query = """
    SELECT * FROM users u, orders o
    WHERE UPPER(u.email) = 'TEST@EXAMPLE.COM'
    AND u.id = o.user_id
    AND o.status NOT IN (SELECT status FROM invalid_statuses)
    AND o.amount = '100'
    GROUP BY u.id, o.order_date
    ORDER BY o.order_date DESC
    """
    
    # Provide table sizes for better cost estimation
    table_sizes = {
        'users': 1000000,
        'orders': 5000000,
        'invalid_statuses': 10
    }
    
    result = suite.analyze_and_optimize(
        problematic_query, 
        auto_fix=True,
        table_sizes=table_sizes
    )
    
    print(f"\n📊 Query Analysis Results:")
    print(f"  Complexity Score: {result['complexity_score']}/100")
    print(f"  Issues Found: {result['issues_found']}")
    print(f"  Cost Level: {result['cost_estimate']['cost_level'].upper()}")
    
    print(f"\n💾 Caching Recommendations:")
    cache = result['caching_recommendations']
    print(f"  Cacheable: {cache['cacheable']}")
    if cache['cacheable']:
        print(f"  Strategy: {cache['cache_strategy']}")
        print(f"  TTL: {cache['ttl_suggestion']}s")
        print(f"  Reasons: {', '.join(cache['reasons'])}")
    
    print(f"\n⚡ Parallel Execution Analysis:")
    parallel = result['parallel_execution']
    print(f"  Parallelizable: {parallel['parallelizable']}")
    if parallel['parallel_opportunities']:
        print(f"  Opportunities:")
        for opp in parallel['parallel_opportunities']:
            print(f"    - {opp}")
    if parallel['estimated_speedup']:
        print(f"  Estimated Speedup: {parallel['estimated_speedup']}")
    
    print(f"\n💰 Cost Estimation:")
    cost = result['cost_estimate']
    print(f"  Total Cost: {cost['total_cost']:.1f} units")
    print(f"  Cost Level: {cost['cost_level']}")
    if cost['primary_cost_drivers']:
        print(f"  Primary Drivers: {', '.join(cost['primary_cost_drivers'])}")
    if cost.get('estimated_credits'):
        print(f"  Snowflake Credits Estimate:")
        for size, credits in cost['estimated_credits'].items():
            print(f"    {size}: {credits} credits")
    
    print(f"\n🔍 Optimization Issues:")
    for issue in result['issues'][:5]:  # Show first 5
        print(f"  [{issue.severity.upper()}] {issue.description}")
        if issue.estimated_impact:
            print(f"    Impact: {issue.estimated_impact}")
    
    if result['optimized_query']:
        print(f"\n✨ Optimized Query:")
        print(result['optimized_query'])
    
    # Example 2: Regression Testing
    print("\n\n2. REGRESSION TESTING EXAMPLE")
    print("-" * 80)
    
    print("""
# To run regression tests with a real database connection:

suite_with_db = QueryOptimizationSuite(
    DatabaseType.POSTGRESQL,
    {
        'host': 'localhost',
        'database': 'testdb',
        'user': 'postgres',
        'password': 'password'
    }
)

suite_with_db.connector.connect()

# Run automated regression test
test_result = suite_with_db.test_optimization(
    original_query=problematic_query,
    optimized_query=optimized_query,
    iterations=5
)

print(f"Test Passed: {test_result['passed']}")
print(f"Performance Improvement: {test_result['comparison']['performance_improvement']:.1f}%")
print(f"Speedup Factor: {test_result['comparison']['speedup_factor']:.2f}x")

# Generate test report
print(suite_with_db.regression_tester.generate_test_report())
    """)
    
    # Example 3: Full Optimization Workflow
    print("\n3. COMPLETE OPTIMIZATION WORKFLOW")
    print("-" * 80)
    
    print("""
# Full end-to-end workflow with analysis, optimization, and testing:

workflow_result = suite.full_optimization_workflow(
    query=your_query,
    test_optimization=True  # Will test if database is connected
)

print("Analysis:", workflow_result['step1_analysis'])
print("Recommendations:", workflow_result['recommendations'])
print("Test Results:", workflow_result['step3_testing'])
    """)
    
    # Example 4: Batch Analysis with All Features
    print("\n4. BATCH FILE ANALYSIS")
    print("-" * 80)
    
    print("""
# Analyze entire SQL file with comprehensive reporting:

results = suite.batch_analyze_file(
    'migrations.sql',
    auto_fix=True,
    output_file='comprehensive_analysis.json'
)

# Results include:
# - Query complexity scores
# - All optimization issues
# - Auto-fixed queries
# - Index recommendations
# - Caching strategies
# - Parallel execution opportunities
# - Cost estimates

for result in results:
    print(f"Query {result['query_hash'][:8]}:")
    print(f"  Complexity: {result['complexity_score']}")
    print(f"  Cost Level: {result['cost_estimate']['cost_level']}")
    print(f"  Cacheable: {result['caching_recommendations']['cacheable']}")
    print(f"  Parallelizable: {result['parallel_execution']['parallelizable']}")
    """)
    
    # Example 5: Real-time Monitoring and Analysis
    print("\n5. REAL-TIME PERFORMANCE MONITORING")
    print("-" * 80)
    
    print("""
# Monitor database and analyze slow queries in real-time:

suite_with_db.connector.connect()

# Find and analyze slow queries
slow_queries = suite_with_db.connector.analyze_slow_queries(
    threshold_ms=1000,
    limit=10
)

for sq in slow_queries:
    print(f"\\nAnalyzing slow query (avg time: {sq['avg_time_ms']:.2f}ms)")
    
    # Full analysis
    analysis = suite_with_db.analyze_and_optimize(
        sq['query'],
        auto_fix=True,
        get_execution_plan=True
    )
    
    # Show recommendations
    print(f"Caching: {analysis['caching_recommendations']['cache_strategy']}")
    print(f"Parallel: {analysis['parallel_execution']['parallelizable']}")
    print(f"Cost: {analysis['cost_estimate']['cost_level']}")
    
    # Test optimization
    if analysis['optimized_query']:
        test = suite_with_db.test_optimization(
            sq['query'],
            analysis['optimized_query']
        )
        print(f"Improvement: {test['comparison']['performance_improvement']:.1f}%")

# Monitor performance metrics
metrics = suite_with_db.connector.monitor_performance_realtime(duration_sec=60)
    """)
    
    # Example 6: Cost Estimation for Different Scenarios
    print("\n6. COST ESTIMATION EXAMPLES")
    print("-" * 80)
    
    test_queries = [
        ("Simple lookup", "SELECT * FROM users WHERE id = 123"),
        ("Complex join", "SELECT * FROM users u JOIN orders o ON u.id = o.user_id JOIN products p ON o.product_id = p.id"),
        ("Heavy aggregation", "SELECT category, COUNT(*), AVG(price) FROM products GROUP BY category HAVING COUNT(*) > 100")
    ]
    
    for name, query in test_queries:
        cost = suite.optimizer.cost_estimator.estimate_cost(query, table_sizes)
        print(f"\n{name}:")
        print(f"  Cost Level: {cost['cost_level']}")
        print(f"  Total Cost: {cost['total_cost']:.1f}")
        if cost['optimization_priority']:
            print(f"  Priority: {cost['optimization_priority'][0]}")
    
    print("\n" + "=" * 80)
    print("✨ FEATURES SUMMARY:")
    print("=" * 80)
    print("""
✓ Query Optimization & Auto-Fix
✓ Live Database Connections (PostgreSQL, MySQL, SQL Server, Snowflake)
✓ Execution Plan Analysis
✓ Index Recommendations with SQL generation
✓ Query Complexity Scoring
✓ Caching Strategy Recommendations
✓ Parallel Execution Analysis
✓ Cost Estimation (including Snowflake credits)
✓ Automated Regression Testing
✓ Performance Monitoring
✓ Batch File Analysis
✓ Comprehensive Reporting
    """)
    
    print("\n📦 Installation:")
    print("pip install sqlparse psycopg2-binary pymysql pyodbc snowflake-connector-python")
    print("\n" + "=" * 80)
    