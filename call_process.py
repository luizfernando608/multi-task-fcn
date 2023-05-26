import subprocess
from src.utils import print_sucess
filename = 'main.py'

while True:
 
    p = subprocess.Popen('python3 '+filename, shell=True).wait()
    print_sucess(p)
    
    if p != 0:
        continue
    else:
        break