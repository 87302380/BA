import time

def print_time(start):
        end = time.time()
        overall_time = end - start
        day = overall_time // (24 * 3600)
        overall_time = overall_time % (24 * 3600)
        hour = overall_time // 3600
        overall_time %= 3600
        minutes = overall_time // 60
        overall_time %= 60
        seconds = overall_time
        print("calcutaion time -> %d:%d:%d:%d" % (day, hour, minutes, seconds))