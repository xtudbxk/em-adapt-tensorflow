import os
import sys
import time
import re
import datetime


def find_gpu(commandline,gpus=[0,1,2,3,4,5],only_cup=False):
    '''
    if gpu in gpus is free, then execute the commandline
    '''
    while(True):
        cmd = "nvidia-smi |awk '{print $2}'"
        with os.popen(cmd) as f:
            flag = False
            using_gpus = set([])
            all_gpus = set([])
            for line in f.readlines():
                r = re.findall("^\d+$",line)
                if len(r) > 0:
                    if flag is True:
                        using_gpus.add(int(r[0]))
                    else:
                        all_gpus.add(int(r[0]))
                if 'Processes:' in line:
                    flag = True
        for gpu in gpus:
            if gpu not in using_gpus:
                if gpu in all_gpus or only_cup is True:
                    cmd = commandline % gpu
                    print("now:%s" % datetime.datetime.now().strftime("%y-%m-%d %H:%M:%s"))
                    print("all gpus:%s" % str(all_gpus))
                    print("using gpus:%s" % str(using_gpus))
                    print("start to execute %s" % cmd)
                    os.system(cmd)
                    sys.exit()
        print("now:%s" % datetime.datetime.now().strftime("%y-%m-%d %H:%M:%s"))
        print("all gpus:%s" % str(all_gpus))
        print("using gpus:%s" % str(using_gpus))

        time.sleep(20)

if __name__ == "__main__":
    cmd = "python deeplab.py %d 2>&1|tee result/20180110-6-0"
    find_gpu(cmd,gpus=[0,1,2,3,4,5],only_cup=False)
