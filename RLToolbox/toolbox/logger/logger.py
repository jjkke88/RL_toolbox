import csv
import time

class Logger(object):
    def __init__(self, head):
        self.head = []
        self.file_name = self.get_file_name()
        self.csvfile = file("log/"+self.file_name , 'wb')
        self.csv_writer = csv.writer(self.csvfile)
        self.log_row(head)

    def log_row(self, data):
        self.csv_writer.writerow(data)

    def get_file_name(self):
        file_time = time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime(time.time()))
        file_name = file_time+".csv"
        return file_name

    def __del__(self):
        self.csvfile.close()