import random
import sys

def mkpass(n=1):
    for i in range(n):
        pwd = chr(65+random.randint(0,25))
        for j in range(7):
            pwd += chr(65+32+random.randint(0,25))
        for k in range(4):
            pwd += chr(48+random.randint(0,9))
        print(pwd)

if __name__ == '__main__':
    try:
        n = int(sys.argv[1])
    except:
        n = 1
    mkpass(n)