import time

class profiler:
    def __init__(self):
        self.time_elapsed = 0
        self.start_time = 0
        self.trigger = -1
    def start(self):
        if self.trigger == 1 :
            print("Call stop!")
        else:
            self.start_time = time.time()
            self.trigger = 1
    def stop(self):
        if self.trigger != 1:
            print("Call start!")
        else:
            time_elapsed = time.time() - self.start_time
            #print("{}: Time elapsed: {}s".format(text, time_elapsed))
            self.trigger = 0
            return time_elapsed


