import os
import sys
import time

cmd = 'nohup ./run_scripts/train.sh&>log.txt'


def gpu_info(gpu_index):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(gpu_status[1].split()[-3][:-1])
    memory = int(gpu_status[2].split('/')[0].strip()[:-3])
    return power, memory    


def narrow_setup(interval=2):
    id = [0,1,2,3]
    for gpu_id in id:
        gpu_power, gpu_memory = gpu_info(gpu_id)
        i = 0
        while gpu_memory > 1000 :  # set waiting condition
            gpu_power, gpu_memory = gpu_info(gpu_id)
            i = i % 5
            symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
            gpu = 'gpu id:%d' % gpu_id
            gpu_power_str = 'gpu power:%d W |' % gpu_power
            gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
            sys.stdout.write('\r' + gpu + ' ' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
            sys.stdout.flush()
            time.sleep(interval)
            i += 1
    os.system(cmd)


if __name__ == '__main__':
    narrow_setup()