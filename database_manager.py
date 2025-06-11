import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
from typing_extensions import (
    List,
    Dict,
    Any,
)
import logging

class Logger:
    def __call__(self):
        logger_ = logging.getLogger(__name__)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s   %(levelname)s   %(name)s:   %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger_.addHandler(console_handler)
        logger_.setLevel(logging.INFO)
        
        return logger_
logger_factory = Logger()
logger_ = logger_factory()

class DatabaseManager:
    """Handles SQLite database operations for metadata management."""
    
    def __init__(self):
        self.current_db_path = None
        self.connection = None
    
    def connect_database(self, db_path: str) -> bool:
        """Connect to a SQLite database file."""
        try:
            if self.connection:
                self.connection.close()
            
            self.connection = sqlite3.connect(db_path)
            self.current_db_path = db_path
            logger_.info(f"Connected to database: {db_path}")
            return True
        except Exception as e:
            logger_.error(f"Failed to connect to database {db_path}: {str(e)}")
            return False
    
    def get_table_data(self, table_name: str = "files_demo") -> pd.DataFrame:
        """Retrieve all data from specified table and return as DataFrame."""
        if not self.connection:
            return pd.DataFrame()
        
        try:
            # Get table data - using your schema column names
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, self.connection)
            return df
        except Exception as e:
            logger_.error(f"Error reading table {table_name}: {str(e)}")
            return pd.DataFrame()
    
    def get_table_schema(self, table_name: str = "files_demo") -> List[str]:
        """Get column names for the specified table."""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [column[1] for column in cursor.fetchall()]
            return columns
        except Exception as e:
            logger_.error(f"Error getting schema for {table_name}: {str(e)}")
            return []
    
    def add_entry(self, entry_data: Dict[str, Any], table_name: str = "files_demo") -> bool:
        """Add a new entry to the database."""
        if not self.connection:
            return False
        
        try:
            # Get current timestamp for lastUpdated
            entry_data['lastUpdated'] = datetime.now().isoformat()
            
            # Build INSERT query dynamically based on provided data
            columns = list(entry_data.keys())
            placeholders = ['?' for _ in columns]
            values = list(entry_data.values())
            
            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            
            cursor = self.connection.cursor()
            cursor.execute(query, values)
            self.connection.commit()
            
            logger_.info(f"Added new entry to {table_name}")
            return True
        except Exception as e:
            logger_.error(f"Error adding entry: {str(e)}")
            return False
    
    def extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata from uploaded file."""
        file_path_obj = Path(file_path)
        
        # Determine modality based on file extension
        modality_mapping = {
            '.mp4': 'video',
            '.wav': 'audio', 
            '.mp3': 'audio',
            '.txt': 'text',
            '.pdf': 'document'
        }
        
        modality = modality_mapping.get(file_path_obj.suffix.lower(), 'unknown')
        
        metadata = {
            'remote_path': str(file_path_obj),
            'modality': modality,
            'toollist': '',  # User will fill this
            'task': '',      # User will fill this
            't_start': 0.0,  # Default values
            't_end': 0.0,
            'perspective': '', # User will fill this
        }
        
        return metadata