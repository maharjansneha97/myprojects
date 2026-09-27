import os
import time

class Logger:
    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, level, msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] [{level}] {msg}"
        print(full_msg)
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(full_msg + "\n")
            except:
                pass

    def info(self, msg):
        self.log("INFO", msg)
    
    def error(self, msg):
        self.log("ERROR", msg)
    
    def success(self, msg):
        self.log("SUCCESS", msg)
    
    def warning(self, msg):
        self.log("WARNING", msg)


class PerformanceMonitor:
    def __init__(self, name="Process"):
        self.name = name
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self, label=None):
        if self.start_time:
            duration = time.time() - self.start_time
            display_name = label if label else self.name
            print(f"[PERF] {display_name} took {duration:.2f}s")
            return duration
        return 0
