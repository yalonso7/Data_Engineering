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
"""

import re
import sqlparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import hashlib


class DatabaseType(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLSERVER = "sqlserver"
    SNOWFLAKE = "snowflake"


@dataclass
class OptimizationIssue:
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str
    description: str
    recommendation: str
    query_section: Optional[str] = None
    auto_fixable: bool = False
    optimized_query: Optional[str] = None
    estimated_impact: Optional[str] = None  # Performance improvement estimate


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
        
    def analyze_file(self, filepath: str, auto_fix: bool = False) -> List[QueryAnalysis]:
        """Analyze SQL file and return comprehensive analysis"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        queries = self._split_queries(content)
        all_analyses = []
        
        for query in queries:
            analysis = self.analyze_query_comprehensive(query, auto_fix=auto_fix)
            all_analyses.append(analysis)
        
        return all_analyses
    
    def analyze_query_comprehensive(self, query: str, auto_fix: bool = False) -> QueryAnalysis:
        """Comprehensive query analysis with all features"""
        # Normalize query for comparison
        normalized = self._normalize_query(query)
        query_hash = hashlib.md5(normalized.encode()).hexdigest()
        
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
            optimized_query=optimized_query
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
        """Analyze all queries in a file and optionally save results"""
        
        analyses = self.optimizer.analyze_file(filepath, auto_fix=auto_fix)
        results = []
        
        for analysis in analyses:
            index_recs = self.optimizer.index_recommender.analyze_query_for_indexes(
                analysis.original_query
            )
            
            results.append({
                'query': analysis.original_query,
                'query_hash': analysis.query_hash,
                'complexity_score': analysis.complexity_score,
                'issues': [
                    {
                        'severity': issue.severity,
                        'category': issue.category,
                        'description': issue.description,
                        'recommendation': issue.recommendation,
                        'auto_fixable': issue.auto_fixable
                    }
                    for issue in analysis.issues
                ],
                'optimized_query': analysis.optimized_query,
                'index_recommendations': index_recs
            })
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
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
    print("SQL Optimization Tool - Complete Enterprise Edition")
    print("=" * 80)
    
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
    