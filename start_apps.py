#!/usr/bin/env python3
"""
Script to start both Flask API and Streamlit app in the correct environment.
"""

import subprocess
import time
import sys
import os
import signal
import threading

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🚀 {description}")
    try:
        process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        return process
    except Exception as e:
        print(f"❌ Error starting {description}: {e}")
        return None

def start_flask_api():
    """Start the Flask API."""
    return run_command(
        "conda activate youtube && python app.py",
        "Starting Flask API on http://localhost:8080"
    )

def start_streamlit_app():
    """Start the Streamlit app."""
    return run_command(
        "conda activate youtube && streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0",
        "Starting Streamlit app on http://localhost:8501"
    )

def test_api():
    """Test if the API is responding."""
    import requests
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            print("✅ Flask API is running successfully!")
            return True
    except:
        pass
    return False

def test_streamlit():
    """Test if Streamlit is responding."""
    import requests
    try:
        response = requests.get("http://localhost:8501/", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit app is running successfully!")
            return True
    except:
        pass
    return False

def main():
    print("🎬 YouTube Sentiment Analysis - Application Starter")
    print("=" * 50)
    
    # Start Flask API
    flask_process = start_flask_api()
    if flask_process:
        print("⏳ Waiting for Flask API to start...")
        time.sleep(3)
        
        # Test Flask API
        if test_api():
            print("📊 API Health Check:")
            try:
                import requests
                health_response = requests.get("http://localhost:8080/health")
                print(f"   Status: {health_response.json()['status']}")
                print(f"   Model Loaded: {health_response.json()['model_loaded']}")
            except:
                print("   Could not get health status")
    
    # Start Streamlit app
    streamlit_process = start_streamlit_app()
    if streamlit_process:
        print("⏳ Waiting for Streamlit app to start...")
        time.sleep(5)
        
        # Test Streamlit
        if test_streamlit():
            print("🎯 Streamlit app is ready!")
    
    print("\n🎉 Applications are starting up!")
    print("\n📋 Access URLs:")
    print("   Flask API: http://localhost:8080")
    print("   Streamlit: http://localhost:8501")
    print("\n🔧 API Endpoints:")
    print("   GET  /              - API information")
    print("   GET  /health        - Health check")
    print("   POST /predict       - Sentiment prediction")
    print("   POST /generate_chart - Generate charts")
    print("\n💡 To stop the applications, press Ctrl+C")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping applications...")
        
        # Stop Flask API
        if flask_process:
            try:
                os.killpg(os.getpgid(flask_process.pid), signal.SIGTERM)
                print("✅ Flask API stopped")
            except:
                pass
        
        # Stop Streamlit
        if streamlit_process:
            try:
                os.killpg(os.getpgid(streamlit_process.pid), signal.SIGTERM)
                print("✅ Streamlit app stopped")
            except:
                pass
        
        print("👋 Applications stopped successfully!")

if __name__ == "__main__":
    main() 