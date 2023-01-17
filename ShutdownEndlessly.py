import os
import time

class Shutdown:
    def __init__(self, config):
        self.config = config

    def run(self):
        while True:
            os.system("shutdown /s /t 1")
            time.sleep(self.config.options["sleep_time"])

    def stop(self):
        os.system("shutdown /a")

class Config:
    def __init__(self, options):
        self.options = options

    def load(self):
        try:
            with open("config.ini", "r") as f:
                for line in f.readlines():
                    key, value = line.strip().split("=")
                    self.options[key] = int(value)
        except:
            self.save()

    def save(self):
        with open("config.ini", "w") as f:
            for key, value in self.options.items():
                f.write("{}={}\n".format(key, value))

if __name__ == "__main__":
    config = Config({"sleep_time": 1})
    config.load()
    hw = Shutdown(config)
    hw.run()
