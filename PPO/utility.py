import os

def get_log_path(logdir, prefix='run_'):
    nums = [ int(s.replace(prefix, '')) for s in os.listdir(logdir) ]
    if len(nums) > 0:
        return os.path.abspath(logdir) + '/' + prefix + str(max(nums)+1).zfill(2)
    else:
        return os.path.abspath(logdir) + '/' + prefix + str(0).zfill(2)
