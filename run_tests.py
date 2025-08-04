#!/usr/bin/env python3
"""
Comprehensive test runner for YouTube Sentiment Analysis project.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

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

def run_tests(test_type="all"):
    """Run tests based on type."""
    commands = {
        "all": "pytest tests/ -v --cov=app --cov=streamlit_app --cov=src --cov-report=html --cov-report=term",
        "unit": "pytest tests/ -v -m unit",
        "api": "pytest tests/test_api.py -v",
        "model": "pytest tests/test_model.py -v",
        "preprocessing": "pytest tests/test_preprocessing.py -v",
        "streamlit": "pytest tests/test_streamlit.py -v",
        "fast": "pytest tests/ -v -x",
        "coverage": "pytest tests/ --cov=app --cov=streamlit_app --cov=src --cov-report=html --cov-report=term-missing",
        "parallel": "pytest tests/ -n auto"
    }
    
    if test_type not in commands:
        print(f"❌ Unknown test type: {test_type}")
        print(f"Available types: {', '.join(commands.keys())}")
        return False
    
    return run_command(commands[test_type], f"Running {test_type} tests")

def run_linting():
    """Run code linting."""
    commands = [
        ("black src/ tests/ app.py streamlit_app.py --check", "Code formatting check (black)"),
        ("isort src/ tests/ app.py streamlit_app.py --check-only", "Import sorting check (isort)"),
        ("flake8 src/ tests/ app.py streamlit_app.py --max-line-length=88", "Linting check (flake8)"),
        ("mypy src/ app.py streamlit_app.py --ignore-missing-imports", "Type checking (mypy)")
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed

def run_security_checks():
    """Run security checks."""
    return run_command(
        "bandit -r . -f json -o bandit-report.json --exclude tests/",
        "Security vulnerability scan (bandit)"
    )

def setup_environment():
    """Setup testing environment."""
    commands = [
        ("pip install -r requirements.txt", "Installing all dependencies"),
        ("pre-commit install", "Installing pre-commit hooks"),
        ("python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')\"", "Downloading NLTK data")
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed

def generate_report():
    """Generate test report."""
    report_file = "test_report.md"
    
    with open(report_file, "w") as f:
        f.write("# Test Report\n\n")
        f.write("## Test Coverage\n")
        f.write("- Unit tests for preprocessing functions\n")
        f.write("- API endpoint tests with validation\n")
        f.write("- Model functionality tests\n")
        f.write("- Streamlit app tests\n")
        f.write("- Error handling tests\n")
        f.write("- Input validation tests\n\n")
        
        f.write("## Code Quality\n")
        f.write("- Black code formatting\n")
        f.write("- Isort import sorting\n")
        f.write("- Flake8 linting\n")
        f.write("- MyPy type checking\n")
        f.write("- Bandit security scanning\n\n")
        
        f.write("## Test Commands\n")
        f.write("```bash\n")
        f.write("# Run all tests\n")
        f.write("python run_tests.py --type all\n\n")
        f.write("# Run specific test types\n")
        f.write("python run_tests.py --type unit\n")
        f.write("python run_tests.py --type api\n")
        f.write("python run_tests.py --type model\n\n")
        f.write("# Run with coverage\n")
        f.write("python run_tests.py --type coverage\n\n")
        f.write("# Run linting\n")
        f.write("python run_tests.py --lint\n\n")
        f.write("# Run security checks\n")
        f.write("python run_tests.py --security\n\n")
        f.write("# Setup environment\n")
        f.write("python run_tests.py --setup\n")
        f.write("```\n")
    
    print(f"📄 Test report generated: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Test runner for YouTube Sentiment Analysis")
    parser.add_argument("--type", default="all", 
                       choices=["all", "unit", "api", "model", "preprocessing", "streamlit", "fast", "coverage", "parallel"],
                       help="Type of tests to run")
    parser.add_argument("--lint", action="store_true", help="Run code linting")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--setup", action="store_true", help="Setup testing environment")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    
    args = parser.parse_args()
    
    print("🧪 YouTube Sentiment Analysis Test Runner")
    print("=" * 50)
    
    if args.setup:
        success = setup_environment()
        if not success:
            sys.exit(1)
    
    if args.lint:
        success = run_linting()
        if not success:
            sys.exit(1)
    
    if args.security:
        success = run_security_checks()
        if not success:
            sys.exit(1)
    
    if args.report:
        generate_report()
    
    if not any([args.lint, args.security, args.setup, args.report]):
        # Run tests by default
        success = run_tests(args.type)
        if not success:
            sys.exit(1)
    
    print("\n🎉 All checks completed!")

if __name__ == "__main__":
    main() 