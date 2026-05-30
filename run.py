import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cleaner_ai import main

if __name__ == '__main__':
    main()
