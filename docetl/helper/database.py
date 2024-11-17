from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple

from rich import Console

from docetl.utils import classproperty


class DatabaseUtil(ABCMeta):
    """
    A utility class for connecting to databases.
    """

    def __init__(self, host: str, port: int, username: str, password: str, database: str,
                 console: Optional[Console] = None):
        self.client = self.new_connection(host=host, port=port, username=username, password=password, database=database,
                                          console=console)
        self.console = console

    def __enter__(self):
        self.console.log("[bold green]Connected to database successfully.[/bold green]")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.console.log("[bold green]Closing connection to database.[/bold green]")

    @abstractmethod
    def new_connection(
            self) -> Any:
        """
        Connect to a database using the provided parameters.

        This function connects to a database using the specified parameters
        and returns a connection object.

        Returns:
            Any: A connection object for interacting with the database.
        """
        pass

    @abstractmethod
    def close_connection(self):
        """
        generic function to close the connection
        :return:
        """
        pass

    @abstractmethod
    def get_query(self, query: str) -> Any:
        """
        Get a query from a file.

        This function reads a query from a file and returns it as a string.

        Args:
            query (str): The path to the query file.

        Returns:
            Any: The query as a string.
        """
        pass

    @abstractmethod
    def execute_query(self, query: str, columns: Optional[Tuple[Any, ...]]) -> Any:
        """
        Perform a query on a SQLite database.

        This function executes a query on a SQLite database and returns the results.

        Args:
            query (str): The SQL query to execute.
            columns (Tuple, optional): The columns to return. Defaults to None.

        Returns:
            Any: The results of the query.
        """
        pass

    @abstractmethod
    def execute_transaction(self, queries: list):
        """
        Execute a list of queries in a transaction

        Args:
            queries: list of queries to execute
        """
        pass

    @abstractmethod
    def log_to_db(self, table_name: str, schema: dict, log_data: str):
        """
        Log data to a database.
        :param table_name:
        :param schema:
        :param log_data:
        :return:
        """
        pass

    @abstractmethod
    def create_log_table(self, table_name: str, schema: dict) -> Any:
        """
        Create a log table in a database.
        :param table_name:
        :param schema:
        :return:
        """
        pass

    #  create a log_schema object that returns key value callable pairs
    class DEFAULT_LOG_SCHEMA():
        def __init__(self, process_id: str = "TEXT", operation: str = "TEXT", log_message: str = "TEXT",
                     table_name: str = "docETL_log"):
            self.process_id: str = process_id
            self.operation: str = operation
            self.log_message: str = log_message
            self.table_name: str = table_name

        @property
        def get_default_table_name(self) -> str:
            """
            Get the default table name for a SQLite database.
            :return: table name
            """
            return "docETL_log"

        @property
        def get_default_log_schema(self) -> dict:
            """
            Get the default log schema for a SQLite database.
            :return: json schema
            """
            return {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "process_id": "TEXT",
                "operation": "TEXT",
                "log_message": "TEXT",
                "timestamp": "DATETIME DEFAULT CURRENT_TIMESTAMP"

            }
