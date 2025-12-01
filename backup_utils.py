# backup_utils.py
import os
from datetime import datetime
import zipfile
from cryptography.fernet import Fernet

def create_backup(db_path: str, backup_dir: str, encrypt_key: str = None) -> str:
    """Create a database backup"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f'backup_{timestamp}.zip')
    
    with zipfile.ZipFile(backup_path, 'w') as zipf:
        zipf.write(db_path, os.path.basename(db_path))
        
    if encrypt_key:
        cipher = Fernet(encrypt_key)
        with open(backup_path, 'rb') as f:
            data = f.read()
        encrypted_data = cipher.encrypt(data)
        with open(backup_path, 'wb') as f:
            f.write(encrypted_data)
    
    return backup_path
