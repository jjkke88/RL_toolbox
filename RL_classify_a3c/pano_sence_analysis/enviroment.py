__author__ = 'Air'
import cv2
#import numpy as np
import parameters as pm
import os
import random
import img_lookat as look

PREFIX_PATH = "prefix_path/"

class Enviroment():
    dataset = []

    def __init__(self, csv_file_path):
        file = open(csv_file_path)
        self.dataset = []
        for line in file:
            line.strip('\n')
            path, label = line.split(';')
            self.dataset+=(self.get_all_file_names(path, label))
        random.shuffle(self.dataset)

    def generate_new_scence(self):
        curr_data = random.choice(self.dataset)
        self.curr_path = curr_data[0]
        self.curr_label = int(curr_data[1])

        self.curr_image = cv2.imread(self.curr_path)
        self.curr_x = random.uniform(-3.141592653, 3.141592653)
        self.curr_y = 0
        self.curr_observe = look.lookat(self.curr_image, self.curr_x, self.curr_y, pm.CROP_SIZE, 3.14159 * 100 / 180)
        return self.curr_observe, self.curr_label


    def generate_dataset(self):
        label_counter = [0]*4
        for data in self.dataset:
            path = data[0]
            label = int(data[1])

            image = cv2.imread(path)
            # curr_x = random.uniform(-3.141592653, -3.141592653)
            x = 0
            y = 0
            observe = look.lookat(image, x, y, pm.CROP_SIZE, 3.14159 * 100 / 180)
            label_counter[label] += 1
            name = "./images/"+str(label)+"_"+str(label_counter[label])+".jpg"
            print name
            cv2.imwrite(name,observe)



    def get_all_file_names(self, path, label):
        file_list = os.listdir(path)

        full_file_lists = []
        if file_list:
            for fn in file_list:
                full_file_name = os.path.join(path, fn)
                full_file_lists.append([full_file_name, label])
        return full_file_lists

    def action(self, act):
        self.curr_x += act
        if(self.curr_x > 3.141592653):
            self.curr_x = -3.141592653 + (self.curr_x - 3.141592653)
        if(self.curr_x < 3.141592653):
            self.curr_x = 3.141592653 - (- 3.141592653 - self.curr_x)
        self.curr_observe = look.lookat(self.curr_image, self.curr_x, self.curr_y, pm.CROP_SIZE, 3.14159 * 100 / 180)
        return self.curr_observe, self.curr_label



