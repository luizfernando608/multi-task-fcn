import subprocess

from src.utils import print_sucess

filename = 'main.py'

while True:

    p = subprocess.Popen('python3 '+ filename, shell=True).wait()
    
    
    if p != 0:
        
        continue
    
    else:
        print_sucess("Process finished")
        break