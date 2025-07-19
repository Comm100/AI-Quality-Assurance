"""Test package initialization - adds app to Python path."""
import sys
import os

# Add the parent directory to Python path so tests can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 