#!/usr/bin/env python3
"""
Simple startup script for YouTube Sentiment Analyzer
Loads from .env and starts both applications
"""

import os
import subprocess
import sys
import time
from dotenv import load_dotenv

def main():
    """Start both Flask API and Streamlit UI."""
    print("🎬 YouTube Sentiment Analyzer")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check if YouTube API key is loaded
    youtube_api_key = os.getenv('YOUTUBE_API_KEY')
    if youtube_api_key:
        print(f"✅ YouTube API key loaded: {youtube_api_key[:10]}...{youtube_api_key[-4:]}")
    else:
        print("⚠️  No YouTube API key found in .env file")
        print("   Please set YOUTUBE_API_KEY in your .env file")
        return
    
    # Check if secret keys are loaded
    secret_key = os.getenv('SECRET_KEY')
    jwt_secret_key = os.getenv('JWT_SECRET_KEY')
    
    if secret_key and jwt_secret_key:
        print("✅ Secret keys loaded from .env")
    else:
        print("⚠️  Secret keys not found in .env file")
        print("   Please run: python generate_secrets.py")
        return
    
    print("\n🚀 Starting applications...")
    
    # Start Flask API
    print("🌐 Starting Flask API...")
    flask_process = subprocess.Popen([
        sys.executable, 'app.py'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    time.sleep(3)
    
    # Start Streamlit UI
    print("📊 Starting Streamlit UI...")
    streamlit_process = subprocess.Popen([
        sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
        '--server.port', '8501',
        '--server.address', 'localhost'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    time.sleep(3)
    
    print("\n🎉 Applications started successfully!")
    print("\n📍 Access Points:")
    print("   • Flask API: http://localhost:8080")
    print("   • Streamlit UI: http://localhost:8501")
    print("   • API Docs: http://localhost:8080/docs")
    print("   • Health Check: http://localhost:8080/health")
    
    print("\n⏹️ Press Ctrl+C to stop both applications")
    
    try:
        while True:
            time.sleep(1)
            
            if flask_process.poll() is not None:
                print("❌ Flask API stopped unexpectedly")
                break
                
            if streamlit_process.poll() is not None:
                print("❌ Streamlit UI stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping applications...")
        
        if flask_process:
            flask_process.terminate()
            print("✅ Flask API stopped")
            
        if streamlit_process:
            streamlit_process.terminate()
            print("✅ Streamlit UI stopped")
        
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 