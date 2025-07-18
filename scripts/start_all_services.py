#!/usr/bin/env python3
"""Script to start all services for development."""
import subprocess
import sys
import time
import signal
import os
from concurrent.futures import ThreadPoolExecutor


def run_service(command: str, name: str, cwd: str = None):
    """Run a service in a subprocess.
    
    Args:
        command: Command to run.
        name: Service name for logging.
        cwd: Working directory.
    """
    print(f"Starting {name}...")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output with service name prefix
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                print(f"[{name}] {line.strip()}")
        
        process.wait()
        
    except KeyboardInterrupt:
        print(f"\n{name} interrupted by user")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"Error running {name}: {e}")


def main():
    """Start all services."""
    print("🚀 Starting AI Quality Assurance Development Environment")
    print("=" * 60)
    
    # Get the project root directory (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Service configurations
    services = [
        {
            "name": "Chat Data Service",
            "command": "python main.py",
            "cwd": os.path.join(project_root, "dummy_services", "chat_data_service")
        },
        {
            "name": "QA Analysis Service",
            "command": "PYTHONPATH=. python app/main.py",
            "cwd": os.path.join(project_root, "qa_analysis_service")
        }
    ]
    
    # Start services in parallel
    try:
        with ThreadPoolExecutor(max_workers=len(services)) as executor:
            futures = []
            
            for service in services:
                future = executor.submit(
                    run_service,
                    service["command"],
                    service["name"],
                    service["cwd"]
                )
                futures.append(future)
            
            print("\n📋 Services starting up...")
            print("   • Chat Data Service: http://localhost:8001")
            print("   • QA Analysis Service: http://localhost:8000")
            print("   • Real RAG Service: https://mqapi.testing.comm100dev.io/...")
            print("\nPress Ctrl+C to stop all services\n")
            
            # Wait for all services
            for future in futures:
                future.result()
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all services...")
        # Cleanup is handled in individual service functions
    
    print("✅ All services stopped")


if __name__ == "__main__":
    main() 