import subprocess

from src.utils import print_sucess

filename = 'main.py'

while True:

    # run this code in terminal
    # sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
    clear_command = 'sync && echo 3 | sudo tee /proc/sys/vm/drop_caches'
    p = subprocess.Popen(clear_command, shell=True).wait()
    
    # alias freecachemem='sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null'
    clear_command_alias = "alias freecachemem='sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null'"
    p = subprocess.Popen(clear_command_alias, shell=True).wait()

 
    p = subprocess.Popen('python3 '+filename, shell=True).wait()
    print_sucess(p)
    
    if p != 0:
        continue
    
    else:
        break