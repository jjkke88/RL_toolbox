from RL_classify.single_step.parameters import PMS_base as pms
from RL_classify.pano_sence_analysis.enviroment import Enviroment
import multiprocess
import os
import cv2
import Queue
class DealImage(multiprocess.Process):
    def __int__(self, task_queue, env, all_class, train_file):
        multiprocess.Process.__init__(self)
        self.task_queue = task_queue
        self.env = env
        self.all_class = all_class
        self.train_file = train_file

    def run(self):
        image_path = "/home/wyp/RL_toolbox/RL_classify/single_step/dataset/JPEGImages/"
        while True:
            task = self.task_queue.get()
            if task != None:
                view , label = self.env.generate_new_scence()
                cv2.imwrite(image_path + str(task) + ".jpg", view)
                self.train_file.write(str(label) + "," + all_class[label-1] + "\n")

if __name__ == "__main__":
    train_env = Enviroment(pms.train_file)
    test_env = Enviroment(pms.test_file)
    all_class = os.listdir("/home/wyp/RL_toolbox/RL_classify/data/SUN360_panoramas_1024x512/train")
    all_class.remove('train.txt')
    print all_class
    producer = []
    train_file = open("/home/wyp/RL_toolbox/RL_classify/single_step/dataset/ImageSets/Main/val.txt", 'w')
    train_file.write(str(len(all_class)))
    for i in xrange(len(all_class)):
        if all_class[i] != "train.txt":
            train_file.write("," + all_class[i])
    train_file.write("\n")
    # for i in xrange(4):
    #     producer.append(DealImage(task_queue , train_env , all_class , train_file))
    image_path = "/home/wyp/RL_toolbox/RL_classify/single_step/dataset/JPEGImages/"
    for i in xrange(10000):
        view, label = train_env.generate_new_scence()
        cv2.imwrite(image_path + str(i) + ".jpg" , view)
        train_file.write(str(i) + ".jpg" + "," + all_class[label - 1] + "\n")
    train_file.close()