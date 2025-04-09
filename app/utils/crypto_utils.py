import base64
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from app.config.env_config import config

def encrypt_password(password: str) -> str:
    """
    Encrypt a password using AES.
    
    Args:
        password: The password to encrypt.
        
    Returns:
        Encrypted password as a base64 string.
    """
    key = config.jwt_private_key.encode()
    iv = os.urandom(16)
    
    # Ensure the key is the right length for AES-256
    key = base64.urlsafe_b64encode(key)[:32].ljust(32, b'=')
    
    # Create a padder for the data
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(password.encode()) + padder.finalize()
    
    # Create an encryptor
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    
    # Encrypt the data
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
    # Combine IV and ciphertext for storage
    result = base64.b64encode(iv + ciphertext).decode('utf-8')
    return result

def decrypt_password(encrypted_password: str) -> str:
    """
    Decrypt an encrypted password.
    
    Args:
        encrypted_password: The encrypted password as a base64 string.
        
    Returns:
        The decrypted password.
    """
    key = config.jwt_private_key.encode()
    
    # Ensure the key is the right length for AES-256
    key = base64.urlsafe_b64encode(key)[:32].ljust(32, b'=')
    
    # Decode the encrypted text
    encrypted_data = base64.b64decode(encrypted_password)
    
    # Extract the IV and ciphertext
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    
    # Create a decryptor
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    
    # Decrypt the data
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    
    # Remove padding
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    
    return data.decode('utf-8') 