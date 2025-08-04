#!/usr/bin/env python3
"""
Dependency installer for YouTube Sentiment Analysis project.
Allows users to install dependencies based on their needs.
"""

import subprocess
import sys
import argparse
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def install_production_deps():
    """Install only production dependencies."""
    production_deps = [
        "Flask==3.0.3",
        "Flask-Cors==5.0.0",
        "streamlit==1.28.1",
        "joblib==1.4.2",
        "lightgbm==4.5.0",
        "scikit-learn==1.5.1",
        "numpy==2.1.2",
        "pandas==2.2.3",
        "nltk==3.9.1",
        "mlflow==2.17.0",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "wordcloud==1.9.3",
        "plotly==5.17.0",
        "dvc==3.53.0",
        "dvc[s3]"
    ]
    
    print("📦 Installing production dependencies...")
    for dep in production_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    return True

def install_all_deps():
    """Install all dependencies from requirements.txt."""
    return run_command("pip install -r requirements.txt", "Installing all dependencies")

def install_dev_tools():
    """Install development tools."""
    dev_tools = [
        "pre-commit",
        "black",
        "isort",
        "flake8",
        "mypy",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pytest-xdist",
        "coverage"
    ]
    
    print("🛠️ Installing development tools...")
    for tool in dev_tools:
        if not run_command(f"pip install {tool}", f"Installing {tool}"):
            return False
    return True

def setup_pre_commit():
    """Setup pre-commit hooks."""
    return run_command("pre-commit install", "Installing pre-commit hooks")

def download_nltk_data():
    """Download required NLTK data."""
    nltk_script = """
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
print("NLTK data downloaded successfully!")
"""
    
    with open("temp_nltk_download.py", "w") as f:
        f.write(nltk_script)
    
    success = run_command("python temp_nltk_download.py", "Downloading NLTK data")
    
    # Clean up
    if os.path.exists("temp_nltk_download.py"):
        os.remove("temp_nltk_download.py")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Install dependencies for YouTube Sentiment Analysis")
    parser.add_argument("--type", default="all", 
                       choices=["production", "development", "all", "dev-tools"],
                       help="Type of installation")
    parser.add_argument("--no-nltk", action="store_true", 
                       help="Skip NLTK data download")
    parser.add_argument("--no-pre-commit", action="store_true", 
                       help="Skip pre-commit setup")
    
    args = parser.parse_args()
    
    print("📦 YouTube Sentiment Analysis - Dependency Installer")
    print("=" * 50)
    
    success = True
    
    if args.type == "production":
        success = install_production_deps()
    elif args.type == "development":
        success = install_dev_tools()
        if success and not args.no_pre_commit:
            success = setup_pre_commit()
    elif args.type == "dev-tools":
        success = install_dev_tools()
    else:  # all
        success = install_all_deps()
        if success and not args.no_pre_commit:
            success = setup_pre_commit()
    
    if success and not args.no_nltk:
        success = download_nltk_data()
    
    if success:
        print("\n🎉 Installation completed successfully!")
        print("\n📋 Next steps:")
        if args.type in ["development", "all"]:
            print("  - Run tests: python run_tests.py")
            print("  - Format code: black .")
            print("  - Check code quality: flake8 .")
        print("  - Start the app: python app.py")
        print("  - Start Streamlit: streamlit run streamlit_app.py")
    else:
        print("\n❌ Installation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 