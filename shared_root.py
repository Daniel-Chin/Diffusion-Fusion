import os
from os import path
from socket import socket

import torch

PORT = 2351
sock = socket()
try:
    sock.bind(('localhost', PORT))
except OSError:
    is_first = False
else:
    is_first = True

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
CPU  = torch.device("cpu")
if HAS_CUDA:
    DEVICE = CUDA
    DEVICE_STR = 'cuda'
    if is_first:
        print('We have CUDA.')
else:
    DEVICE = CPU
    DEVICE_STR = 'cpu'
    if is_first:
        print("We DON'T have CUDA.")

if __name__ == '__main__':
    SHARED_ROOT = 'shared_root.py'
    EXCLUDE = ['.git', '__pycache__']

    def main():
        assert path.basename(__file__) == SHARED_ROOT
        with open(__file__, 'r') as f:
            data = f.read()
        propagate(data)
        print('ok')

    def propagate(data, depth=0, display='.'):
        with open('shared.py', 'w') as f:
            print('# Auto generated by', __file__, file=f)
            print(file=f)
            f.write(data)
        print(' ' * depth, display, sep='')
        for name in os.listdir():
            if path.isdir(name) and name not in EXCLUDE:
                os.chdir(name)
                propagate(data, depth + 1, name)
                os.chdir('..')

    main()
