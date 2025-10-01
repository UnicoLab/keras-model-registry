"""Configure pytest for the project.

This file ensures that imports and paths are properly set up for testing.
"""

import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
