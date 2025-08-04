#!/usr/bin/env python3
"""
Generate secure secret keys for the YouTube Comment Intelligence
"""
import secrets
import string

def generate_secret_key(length=32):
    """Generate a secure secret key."""
    return secrets.token_hex(length)

def generate_jwt_secret_key(length=32):
    """Generate a secure JWT secret key."""
    return secrets.token_hex(length)

def generate_api_key(length=32):
    """Generate a secure API key."""
    # Use alphanumeric characters for API keys
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

def main():
    """Generate all secret keys."""
    print("🔐 Generating Secure Secret Keys")
    print("=" * 40)
    
    # Generate secret keys
    secret_key = generate_secret_key(32)
    jwt_secret_key = generate_jwt_secret_key(32)
    api_key = generate_api_key(32)
    
    print(f"SECRET_KEY={secret_key}")
    print(f"JWT_SECRET_KEY={jwt_secret_key}")
    print(f"API_KEY={api_key}")
    
    print("\n📋 Copy these to your .env file:")
    print("=" * 40)
    print(f"SECRET_KEY={secret_key}")
    print(f"JWT_SECRET_KEY={jwt_secret_key}")
    
    print("\n🔐 Security Notes:")
    print("   • Keep these keys secret and secure")
    print("   • Never commit them to version control")
    print("   • Use different keys for development and production")
    print("   • Rotate keys regularly for security")

if __name__ == "__main__":
    main() 