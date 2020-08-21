import os
from glob import glob
path = r'G:\CT_DATA\202001\钙化伪影'
for x in os.listdir(path):
    path1 = os.path.join(path, x)
    for x1 in os.listdir(path1):
        PATH = os.path.join(path1, x1)
        data = glob(os.path.join(PATH,'DS_CORCTA_0_75*20-80*/*'))
        label = glob(os.path.join(PATH,'DS_CORCTA_0_75*BEST*/*'))
        ldata = len(data)
        llabel = len(label)
        if ldata != llabel:
            print(PATH)