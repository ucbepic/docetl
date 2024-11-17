import sqlite3
from typing import Optional, Tuple, Any

import rich
from rich import Console
from typing_extensions import override

from docetl.helper.database import DatabaseUtil


class SqliteUtil(DatabaseUtil):
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.console = rich.Console()

    @override
    def new_connection(self):
        if self.conn is None:
            try:
                self.conn = sqlite3.connect(self.db_path)
                self.cursor = self.conn.cursor()
                self.console.log("[bold green]Connected to SQLite database successfully.[/bold green]")
            except Exception as e:
                self.console.log(f"[bold red]Error connecting to SQLite database: {str(e)}[/bold red]")
                raise ConnectionError(f"Error connecting to SQLite database: {str(e)}")

        return self.conn

    @override
    def close_connection(self):
        if self.conn is not None:
            try:
                self.conn.close()
                self.conn = None
                self.cursor = None
                self.console.log("[bold green]Disconnected from SQLite database successfully.[/bold green]")
            except Exception as e:
                self.console.log(f"[bold red]Error disconnecting from SQLite database: {str(e)}[/bold red]")
                raise ConnectionError(f"Error disconnecting from SQLite database: {str(e)}")


    @override
    def get_query(self, query: str) -> Any:
        if self.conn is None:
            raise ConnectionError("No connection to database")
        try:
            self.console.log(f"Executing query: {query}")
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            raise ValueError(f"Error executing query: {str(e)}")

    @override
    def execute_query(self, query: str, columns: Optional[Tuple[Any, ...]]) -> Any:
        if self.conn is None:
            raise ConnectionError("No connection to database")
        try:
            self.console.log(f"Executing query: {query}")
            self.cursor.execute(query, columns)
            return self.conn.commit()
        except Exception as e:
            raise ValueError(f"Error executing query: {str(e)}")

    @override
    def execute_transaction(self, queries: list):
        if self.conn is None:
            raise ConnectionError("No connection to database")
        try:
            for query in queries:
                self.cursor.execute(query)
            self.conn.commit()
            self.console.log("[bold green]Transaction executed successfully.[/bold green]")
        except Exception as e:
            self.conn.rollback()
            self.console.log(f"[bold red]Error executing transaction: {str(e)}[/bold red]")
            raise ValueError(f"Error executing transaction: {str(e)}")

    @override
    def log_to_db(self, table_name: str, schema: dict, log_data: str):
        """
        Log data to a SQLite database.

        This function logs data to a SQLite database.

        Args:
            table_name (str): The name of the table to log to.
            schema (dict): The schema of the table.
            log_data (str): The data to log in schema format.
        """
        for key in schema.keys():
            if key not in DatabaseUtil.DEFAULT_LOG_SCHEMA.get_default_log_schema.keys():
                raise ValueError(f"Column '{key}' is not defined in the schema.")

        columns = ', '.join(schema.keys())  # Get the column names
        placeholders = ', '.join('?' * len(schema))  # Create placeholders for the values
        insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"

        return self.execute_query(insert_sql, tuple(log_data.values()))


    @override
    def create_log_table(self, table_name: str, schema: dict) -> Any:
        """
        Create a log table in a SQLite database.

        This function creates a log table in a SQLite database.

        Args:
            table_name (str): The name of the table to create.
            schema (dict): The schema of the table.
        """
        if not schema:
            schema = DatabaseUtil.DEFAULT_LOG_SCHEMA.get_default_log_schema

        if not table_name:
            table_name = DatabaseUtil.DEFAULT_LOG_SCHEMA.get_default_table_name

        columns = ", ".join([f"{column} {datatype}" for column, datatype in schema.items()])

        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
        self.execute_query(query=create_table_sql)
