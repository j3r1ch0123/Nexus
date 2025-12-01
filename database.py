# database.py
import sqlite3
import os
import json
from datetime import datetime
import re
import zipfile
from typing import List, Dict
from cryptography.fernet import Fernet

# database.py
class ResearchDatabase:
    def __init__(self, db_path: str = 'research.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_table()  # Initialize the table on creation
        
    def _create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS research_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            content TEXT,
            source TEXT,
            published_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.conn.execute(query)
        self.conn.commit()
        
    def migrate(self):
        """Handle database schema migrations"""
        version = self._get_db_version()
        
        if version < 1:
            self._create_table()
            self._set_db_version(1)
            
        if version < 2:
            self._add_indexes()
            self._set_db_version(2)
    
    def backup(self, backup_path: str, encrypt_key: str = None):
        """Create a database backup"""
        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'data': self.export_all()
        }
        
        if encrypt_key:
            cipher = Fernet(encrypt_key)
            backup_data = cipher.encrypt(json.dumps(backup_data).encode())
        else:
            backup_data = json.dumps(backup_data).encode()
            
        with open(backup_path, 'wb') as f:
            f.write(backup_data)
    
    def export_all(self) -> List[Dict]:
        """Export all research results"""
        query = "SELECT * FROM research_results"
        cursor = self.conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]
    
    def _add_indexes(self):
        """Add performance indexes"""
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_query ON research_results(query)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON research_results(source)")
        self.conn.commit()
    
    def _get_db_version(self) -> int:
        """Get current database version"""
        try:
            cursor = self.conn.execute("PRAGMA user_version")
            return cursor.fetchone()[0]
        except:
            return 0
    
    def _set_db_version(self, version: int):
        """Set database version"""
        self.conn.execute(f"PRAGMA user_version = {version}")
        self.conn.commit()
