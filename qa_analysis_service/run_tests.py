#!/usr/bin/env python3
"""Comprehensive test runner for QA Analysis Service."""
import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Optional


class TestRunner:
    """Test runner with different test suites and configurations."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent
        # Ensure PYTHONPATH is set for imports
        os.environ['PYTHONPATH'] = str(self.project_root)
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        self.log(f"Running: {description}")
        if self.verbose:
            self.log(f"Command: {' '.join(cmd)}")
        
        try:
            # Create environment with PYTHONPATH set
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=not self.verbose,
                text=True,
                check=False,
                env=env
            )
            
            if result.returncode == 0:
                self.log(f"✅ {description} - PASSED")
                return True
            else:
                self.log(f"❌ {description} - FAILED (exit code: {result.returncode})", "ERROR")
                if not self.verbose and result.stdout:
                    print(result.stdout)
                if not self.verbose and result.stderr:
                    print(result.stderr)
                return False
                
        except Exception as e:
            self.log(f"❌ {description} - ERROR: {e}", "ERROR")
            return False
    
    def smoke_tests(self) -> bool:
        """Run quick smoke tests."""
        self.log("🔥 Running Smoke Tests")
        cmd = ["python", "-m", "pytest", "-m", "smoke", "--tb=line", "-v"]
        return self.run_command(cmd, "Smoke tests")
    
    def unit_tests(self) -> bool:
        """Run unit tests with coverage."""
        self.log("🧪 Running Unit Tests")
        cmd = [
            "python", "-m", "pytest", 
            "-m", "unit",
            "--cov=app",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "-v"
        ]
        return self.run_command(cmd, "Unit tests")
    
    def integration_tests(self) -> bool:
        """Run integration tests."""
        self.log("🔗 Running Integration Tests")
        cmd = [
            "python", "-m", "pytest",
            "-m", "integration",
            "--tb=short",
            "-v"
        ]
        return self.run_command(cmd, "Integration tests")
    
    def api_tests(self) -> bool:
        """Run API contract tests."""
        self.log("🌐 Running API Tests")
        cmd = [
            "python", "-m", "pytest",
            "-m", "api",
            "--tb=short",
            "-v"
        ]
        return self.run_command(cmd, "API tests")
    
    def performance_tests(self) -> bool:
        """Run performance tests."""
        self.log("⚡ Running Performance Tests")
        cmd = [
            "python", "-m", "pytest",
            "-m", "performance",
            "--tb=line",
            "-v",
            "-s"  # Show print statements for performance metrics
        ]
        return self.run_command(cmd, "Performance tests")
    
    def security_tests(self) -> bool:
        """Run security tests."""
        self.log("🔒 Running Security Tests")
        cmd = [
            "python", "-m", "pytest",
            "-m", "security",
            "--tb=short",
            "-v"
        ]
        return self.run_command(cmd, "Security tests")
    
    def all_tests(self) -> bool:
        """Run all tests."""
        self.log("🚀 Running All Tests")
        cmd = [
            "python", "-m", "pytest",
            "--cov=app",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=80",
            "--tb=short",
            "-v"
        ]
        return self.run_command(cmd, "All tests")
    
    def linting(self) -> bool:
        """Run code linting."""
        self.log("📝 Running Code Linting")
        
        # Check if flake8 is available
        try:
            result = subprocess.run(["flake8", "--version"], capture_output=True)
            if result.returncode != 0:
                self.log("⚠️  flake8 not available, skipping linting")
                return True
        except FileNotFoundError:
            self.log("⚠️  flake8 not installed, skipping linting")
            return True
        
        cmd = [
            "flake8",
            "app/",
            "--max-line-length=120",
            "--ignore=E203,W503,E501",
            "--exclude=__pycache__,*.pyc"
        ]
        return self.run_command(cmd, "Code linting")
    
    def type_checking(self) -> bool:
        """Run type checking."""
        self.log("🔍 Running Type Checking")
        
        # Check if mypy is available
        try:
            result = subprocess.run(["mypy", "--version"], capture_output=True)
            if result.returncode != 0:
                self.log("⚠️  mypy not available, skipping type checking")
                return True
        except FileNotFoundError:
            self.log("⚠️  mypy not installed, skipping type checking")
            return True
        
        cmd = [
            "mypy",
            "app/",
            "--ignore-missing-imports",
            "--no-strict-optional"
        ]
        return self.run_command(cmd, "Type checking")
    
    def generate_coverage_report(self) -> bool:
        """Generate detailed coverage report."""
        self.log("📊 Generating Coverage Report")
        
        # Run tests with coverage
        cmd = [
            "python", "-m", "pytest",
            "--cov=app",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--cov-report=term-missing",
            "-q"
        ]
        
        if self.run_command(cmd, "Coverage generation"):
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    data = json.load(f)
                    total_coverage = data.get("totals", {}).get("percent_covered", 0)
                    self.log(f"📈 Total Coverage: {total_coverage:.1f}%")
                    
                    if total_coverage >= 80:
                        self.log("✅ Coverage target met (≥80%)")
                    else:
                        self.log("⚠️  Coverage below target (<80%)", "WARNING")
            
            self.log(f"📄 HTML Coverage Report: {self.project_root}/htmlcov/index.html")
            return True
        
        return False
    
    def health_check(self) -> bool:
        """Run health check tests for production monitoring."""
        self.log("❤️  Running Health Check")
        
        # Basic import test
        try:
            sys.path.append(str(self.project_root))
            from app.config import settings
            from app.main import app
            from app.services.analysis_service import AnalysisService
            
            self.log("✅ All imports successful")
            
            # Configuration validation
            if not settings.openai_api_key:
                self.log("⚠️  OpenAI API key not set", "WARNING")
            else:
                self.log("✅ OpenAI API key configured")
            
            self.log(f"✅ Configuration loaded: {settings.app_name} v{settings.api_version}")
            return True
            
        except Exception as e:
            self.log(f"❌ Health check failed: {e}", "ERROR")
            return False
    
    def ci_pipeline(self) -> bool:
        """Run CI/CD pipeline tests."""
        self.log("🔄 Running CI Pipeline")
        
        steps = [
            ("Health Check", self.health_check),
            ("Linting", self.linting),
            ("Type Checking", self.type_checking),
            ("Smoke Tests", self.smoke_tests),
            ("Unit Tests", self.unit_tests),
            ("Integration Tests", self.integration_tests),
            ("API Tests", self.api_tests),
            ("Coverage Report", self.generate_coverage_report)
        ]
        
        failed_steps = []
        for step_name, step_func in steps:
            if not step_func():
                failed_steps.append(step_name)
        
        if failed_steps:
            self.log(f"❌ CI Pipeline FAILED. Failed steps: {', '.join(failed_steps)}", "ERROR")
            return False
        else:
            self.log("✅ CI Pipeline PASSED")
            return True
    
    def pre_production_tests(self) -> bool:
        """Run comprehensive pre-production test suite."""
        self.log("🚦 Running Pre-Production Tests")
        
        steps = [
            ("Health Check", self.health_check),
            ("All Tests", self.all_tests),
            ("Performance Tests", self.performance_tests),
            ("Security Tests", self.security_tests),
            ("Coverage Report", self.generate_coverage_report)
        ]
        
        failed_steps = []
        for step_name, step_func in steps:
            if not step_func():
                failed_steps.append(step_name)
        
        if failed_steps:
            self.log(f"❌ Pre-Production Tests FAILED. Failed steps: {', '.join(failed_steps)}", "ERROR")
            return False
        else:
            self.log("✅ Pre-Production Tests PASSED - Ready for deployment!")
            return True
    
    def full_report(self) -> bool:
        """Run all tests and generate comprehensive reports in multiple formats."""
        self.log("📊 Running Complete Test Suite with Full Reporting")
        
        # Check if pytest plugins are installed
        try:
            import pytest_html
            import pytest_json_report
        except ImportError:
            self.log("⚠️  Installing missing report plugins...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest-html", "pytest-json-report"], 
                         capture_output=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_dir = self.project_root / f"test-reports-{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        self.log(f"📁 Reports will be saved to: {report_dir}")
        
        cmd = [
            "python", "-m", "pytest",
            # Coverage options
            "--cov=app",
            "--cov-report=term-missing",
            f"--cov-report=html:{report_dir}/coverage-html",
            f"--cov-report=xml:{report_dir}/coverage.xml",
            f"--cov-report=json:{report_dir}/coverage.json",
            "--cov-fail-under=80",
            # Report formats
            f"--html={report_dir}/test-report.html",
            "--self-contained-html",
            f"--junit-xml={report_dir}/junit-report.xml",
            f"--json-report-file={report_dir}/test-results.json",
            # Output options
            "--tb=short",
            "-v",
            "--durations=20",
            "--maxfail=10"
        ]
        
        success = self.run_command(cmd, "Complete test suite with reporting")
        
        if success:
            self.log("✅ All tests completed successfully!")
            self.log(f"📊 Reports generated:")
            self.log(f"   📄 HTML Test Report: {report_dir}/test-report.html")
            self.log(f"   📈 Coverage Report: {report_dir}/coverage-html/index.html")
            self.log(f"   📋 JUnit XML: {report_dir}/junit-report.xml")
            self.log(f"   📊 JSON Report: {report_dir}/test-results.json")
            self.log(f"   📉 Coverage XML: {report_dir}/coverage.xml")
            
            # Print summary from coverage.json
            coverage_json = report_dir / "coverage.json"
            if coverage_json.exists():
                with open(coverage_json) as f:
                    coverage_data = json.load(f)
                    total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                    self.log(f"\n📊 TOTAL COVERAGE: {total_coverage:.1f}%")
                    
                    # Show file-by-file coverage
                    self.log("\n📁 Coverage by file:")
                    files = coverage_data.get("files", {})
                    for file_path, file_data in sorted(files.items()):
                        if "app/" in file_path:
                            coverage = file_data.get("summary", {}).get("percent_covered", 0)
                            self.log(f"   {file_path}: {coverage:.1f}%")
        
        return success


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="QA Analysis Service Test Runner")
    parser.add_argument("--suite", choices=[
        "smoke", "unit", "integration", "api", "performance", "security",
        "all", "health", "ci", "pre-prod", "full-report"
    ], default="smoke", help="Test suite to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    
    # Map suite names to methods
    suite_map = {
        "smoke": runner.smoke_tests,
        "unit": runner.unit_tests,
        "integration": runner.integration_tests,
        "api": runner.api_tests,
        "performance": runner.performance_tests,
        "security": runner.security_tests,
        "all": runner.all_tests,
        "health": runner.health_check,
        "ci": runner.ci_pipeline,
        "pre-prod": runner.pre_production_tests,
        "full-report": runner.full_report
    }
    
    start_time = time.time()
    
    # Run selected test suite
    success = suite_map[args.suite]()
    
    # Generate coverage report if requested
    if args.coverage and args.suite not in ["health", "ci", "pre-prod"]:
        runner.generate_coverage_report()
    
    end_time = time.time()
    duration = end_time - start_time
    
    runner.log(f"⏱️  Total execution time: {duration:.2f} seconds")
    
    if success:
        runner.log("🎉 Test suite completed successfully!")
        sys.exit(0)
    else:
        runner.log("💥 Test suite failed!", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main() 