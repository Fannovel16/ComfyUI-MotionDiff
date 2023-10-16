import os
from pathlib import Path
import sys

s_param = '-s' if "python_embeded" in sys.executable else '' 
with open(Path(__file__).parent / "requirements.txt", 'r') as f:
    for package in f.readlines():
        package = package.strip()
        print(f"Installing {package}...")
        os.system(f'"{sys.executable}" {s_param} -m pip install {package}')
