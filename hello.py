
print('hello youssef ')
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
print( 'ths is the path folder' ,PROJECT_ROOT)


from pathlib import Path

PROJECT_ROOT = Path.cwd()   # ← THIS is the key
print( 'ths is the path folder' ,PROJECT_ROOT)