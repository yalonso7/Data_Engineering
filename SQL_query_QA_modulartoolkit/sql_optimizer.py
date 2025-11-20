"""
SQL Optimization and Performance Monitoring Tool
Supports: PostgreSQL, MySQL, SQL Server, Snowflake
"""

import re
import sqlparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


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


class SQLOptimizer:
    def __init__(self, db_type: DatabaseType = DatabaseType.POSTGRESQL):
        self.db_type = db_type
        self.issues: List[OptimizationIssue] = []
        
    def analyze_file(self, filepath: str) -> List[OptimizationIssue]:
        """Analyze SQL file and return optimization recommendations"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        queries = self._split_queries(content)
        all_issues = []
        
        for query in queries:
            issues = self.analyze_query(query)
            all_issues.extend(issues)
        
        return all_issues
    
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
                auto_fixable=False
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
                    auto_fixable=False
                ))
    
    def _check_non_sargable(self, query: str):
        """Check for non-SARGable predicates that prevent index usage"""
        patterns = [
            (r'\bWHERE\s+\w+\s*\+\s*\d+', 'Arithmetic operation on indexed column'),
            (r'\bWHERE\s+UPPER\(', 'Function on indexed column (UPPER)'),
            (r'\bWHERE\s+LOWER\(', 'Function on indexed column (LOWER)'),
            (r'\bWHERE\s+SUBSTRING\(', 'Function on indexed column (SUBSTRING)'),
            (r'\bWHERE\s+YEAR\(', 'Function on indexed column (YEAR)'),
            (r'\bWHERE\s+CAST\(', 'Type conversion on indexed column'),
        ]
        
        for pattern, desc in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                self.issues.append(OptimizationIssue(
                    severity='high',
                    category='Performance',
                    description=f'Non-SARGable predicate: {desc}',
                    recommendation='Rewrite query to avoid functions on indexed columns',
                    auto_fixable=False
                ))
    
    def _check_implicit_conversions(self, query: str):
        """Check for potential implicit type conversions"""
        # Check for quoted numbers which might cause implicit conversion
        if re.search(r'=\s*["\'][0-9]+["\']', query):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Performance',
                description='Potential implicit type conversion with quoted numeric value',
                recommendation='Use unquoted numbers for numeric columns',
                auto_fixable=True
            ))
    
    def _check_or_conditions(self, query: str):
        """Check for OR conditions that might benefit from UNION"""
        or_count = len(re.findall(r'\bOR\b', query, re.IGNORECASE))
        
        if or_count >= 3:
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Performance',
                description=f'Multiple OR conditions ({or_count}) may prevent index usage',
                recommendation='Consider rewriting as UNION or using IN clause if applicable',
                auto_fixable=False
            ))
    
    def _check_subquery_optimization(self, query: str, statement):
        """Check for subqueries that could be optimized"""
        if re.search(r'\bIN\s*\(\s*SELECT', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Performance',
                description='IN with subquery can be slow for large datasets',
                recommendation='Consider using JOIN or EXISTS instead',
                auto_fixable=False
            ))
        
        if re.search(r'\bNOT\s+IN\s*\(\s*SELECT', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='high',
                category='Performance',
                description='NOT IN with subquery can be very slow',
                recommendation='Use NOT EXISTS or LEFT JOIN with NULL check instead',
                auto_fixable=False
            ))
    
    def _check_distinct_usage(self, query: str):
        """Check for DISTINCT usage"""
        if re.search(r'\bSELECT\s+DISTINCT\b', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='low',
                category='Performance',
                description='DISTINCT requires sorting/hashing of results',
                recommendation='Verify if DISTINCT is necessary; consider fixing data model if used frequently',
                auto_fixable=False
            ))
    
    def _check_join_types(self, query: str):
        """Check for join optimization opportunities"""
        if re.search(r'\bCROSS\s+JOIN\b', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='high',
                category='Performance',
                description='CROSS JOIN creates cartesian product',
                recommendation='Ensure CROSS JOIN is intentional; add join conditions if needed',
                auto_fixable=False
            ))
        
        # Check for comma joins (old style)
        from_match = re.search(r'\bFROM\s+(\w+)\s*,\s*(\w+)', query, re.IGNORECASE)
        if from_match:
            self.issues.append(OptimizationIssue(
                severity='low',
                category='Best Practice',
                description='Using comma-separated tables (implicit join)',
                recommendation='Use explicit JOIN syntax for better readability',
                auto_fixable=True
            ))
    
    def _check_index_hints(self, query: str):
        """Check for missing index opportunities"""
        # Look for WHERE clause columns
        where_cols = re.findall(r'\bWHERE\s+(\w+)', query, re.IGNORECASE)
        join_cols = re.findall(r'\bON\s+\w+\.(\w+)\s*=', query, re.IGNORECASE)
        
        if where_cols or join_cols:
            self.issues.append(OptimizationIssue(
                severity='low',
                category='Information',
                description='Ensure indexes exist on WHERE and JOIN columns',
                recommendation=f'Consider indexes on: {", ".join(set(where_cols + join_cols))}',
                auto_fixable=False
            ))
    
    def _check_like_patterns(self, query: str):
        """Check for inefficient LIKE patterns"""
        if re.search(r'LIKE\s+["\']%', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='high',
                category='Performance',
                description='LIKE with leading wildcard prevents index usage',
                recommendation='Avoid leading wildcards in LIKE patterns when possible',
                auto_fixable=False
            ))
    
    def _check_union_vs_union_all(self, query: str):
        """Check for UNION vs UNION ALL"""
        if re.search(r'\bUNION\s+(?!ALL)', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Performance',
                description='UNION removes duplicates (expensive operation)',
                recommendation='Use UNION ALL if duplicates are acceptable',
                auto_fixable=True
            ))
    
    def _check_aggregation_issues(self, query: str):
        """Check for aggregation performance issues"""
        if re.search(r'\bGROUP\s+BY\b', query, re.IGNORECASE):
            if not re.search(r'\bHAVING\b', query, re.IGNORECASE):
                # Check if WHERE could be used instead of HAVING
                pass
            
            # Count number of columns in GROUP BY
            group_by_match = re.search(r'GROUP\s+BY\s+(.+?)(?:ORDER|HAVING|LIMIT|$)', 
                                      query, re.IGNORECASE | re.DOTALL)
            if group_by_match:
                cols = [c.strip() for c in group_by_match.group(1).split(',')]
                if len(cols) > 5:
                    self.issues.append(OptimizationIssue(
                        severity='medium',
                        category='Performance',
                        description=f'GROUP BY with many columns ({len(cols)}) can be slow',
                        recommendation='Consider if all grouping columns are necessary',
                        auto_fixable=False
                    ))
    
    def _check_null_checks(self, query: str):
        """Check for NULL handling"""
        if re.search(r'=\s*NULL|NULL\s*=', query, re.IGNORECASE):
            self.issues.append(OptimizationIssue(
                severity='medium',
                category='Correctness',
                description='Using = NULL instead of IS NULL',
                recommendation='Use IS NULL or IS NOT NULL for NULL checks',
                auto_fixable=True
            ))
    
    def _check_database_specific(self, query: str):
        """Check for database-specific optimizations"""
        if self.db_type == DatabaseType.POSTGRESQL:
            # Check for inefficient string concatenation
            if re.search(r'\|\|', query):
                self.issues.append(OptimizationIssue(
                    severity='low',
                    category='Best Practice',
                    description='String concatenation in query',
                    recommendation='Consider using CONCAT() or formatting functions',
                    auto_fixable=False
                ))
        
        elif self.db_type == DatabaseType.MYSQL:
            # Check for missing LIMIT in DELETE/UPDATE
            if re.search(r'\b(DELETE|UPDATE)\b', query, re.IGNORECASE):
                if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
                    self.issues.append(OptimizationIssue(
                        severity='low',
                        category='Safety',
                        description='DELETE/UPDATE without LIMIT',
                        recommendation='Consider adding LIMIT clause for safety',
                        auto_fixable=False
                    ))
        
        elif self.db_type == DatabaseType.SNOWFLAKE:
            # Check for clustering key opportunities
            if re.search(r'\bWHERE\s+\w+\s+BETWEEN', query, re.IGNORECASE):
                self.issues.append(OptimizationIssue(
                    severity='low',
                    category='Information',
                    description='Range query detected',
                    recommendation='Consider clustering key on range-queried columns',
                    auto_fixable=False
                ))
    
    def generate_report(self, issues: List[OptimizationIssue]) -> str:
        """Generate a formatted report of issues"""
        if not issues:
            return "✓ No optimization issues found!"
        
        report = ["SQL Optimization Report", "=" * 50, ""]
        
        # Group by severity
        by_severity = {'critical': [], 'high': [], 'medium': [], 'low': []}
        for issue in issues:
            by_severity[issue.severity].append(issue)
        
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_issues = by_severity[severity]
            if not severity_issues:
                continue
            
            report.append(f"\n{severity.upper()} ({len(severity_issues)} issues)")
            report.append("-" * 50)
            
            for i, issue in enumerate(severity_issues, 1):
                report.append(f"\n{i}. [{issue.category}] {issue.description}")
                report.append(f"   → {issue.recommendation}")
                if issue.query_section:
                    report.append(f"   Section: {issue.query_section}")
        
        report.append(f"\n\nTotal issues found: {len(issues)}")
        return "\n".join(report)


class DatabaseConnector:
    """Connect to databases and analyze queries"""
    
    def __init__(self, db_type: DatabaseType, connection_params: Dict[str, Any]):
        self.db_type = db_type
        self.connection_params = connection_params
        self.optimizer = SQLOptimizer(db_type)
    
    def connect(self):
        """Establish database connection"""
        # Implementation would use appropriate driver:
        # - psycopg2 for PostgreSQL
        # - pymysql for MySQL
        # - pyodbc for SQL Server
        # - snowflake-connector-python for Snowflake
        pass
    
    def analyze_slow_queries(self, threshold_ms: int = 1000) -> List[Dict]:
        """Analyze slow queries from database logs"""
        # This would query system tables/views for slow queries:
        # PostgreSQL: pg_stat_statements
        # MySQL: slow_query_log
        # SQL Server: sys.dm_exec_query_stats
        # Snowflake: QUERY_HISTORY view
        pass
    
    def get_execution_plan(self, query: str) -> Dict:
        """Get query execution plan"""
        # Would use EXPLAIN or equivalent for each database
        pass


# Example usage
if __name__ == "__main__":
    # Example 1: Analyze a query
    optimizer = SQLOptimizer(DatabaseType.POSTGRESQL)
    
    sample_query = """
    SELECT * FROM users u, orders o
    WHERE UPPER(u.email) = 'TEST@EXAMPLE.COM'
    AND o.user_id = u.id
    AND o.status NOT IN (SELECT status FROM invalid_statuses)
    """
    
    issues = optimizer.analyze_query(sample_query)
    print(optimizer.generate_report(issues))
    
    # Example 2: Analyze a file
    # issues = optimizer.analyze_file('queries.sql')
    # print(optimizer.generate_report(issues))