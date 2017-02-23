from RLToolbox.agent.TRPO_agent import TRPOAgent
import os
import cv2
import numpy as np

class TrackingAgent(TRPOAgent):
    def __init__(self , env , session , baseline , storage , distribution , net , pms):
        super(TrackingAgent , self).__init__(env , session , baseline , storage , distribution , net , pms)

    def test_tracking(self, checkpoint_file, data_file_path):
        self.load_model(checkpoint_file)
        full_file_lists = self.get_all_file_names(data_file_path)
        image_number = len(full_file_lists)
        print "data image number:%d"%image_number
        zfill_i = str(1).zfill(4)
        image_path = data_file_path + "/" + zfill_i + ".jpg"
        image = cv2.imread(image_path)

        template_position = (83,200,35 + 85,131 + 200) #left top, right bottom

        template_image = image[template_position[0]:template_position[2] ,
                         template_position[1]:template_position[3] , :]

        for i in xrange(image_number-1):
            image_clone = image.copy();
            cv2.rectangle(image_clone , (template_position[0] , template_position[1]) ,
                          (template_position[2] , template_position[3]) , (255 , 0 , 0))
            cv2.imshow("tracking" , image_clone)
            cv2.waitKey()
            zfill_i = str(i+2).zfill(4)
            image_path = data_file_path + "/" + zfill_i + ".jpg"
            image = cv2.imread(image_path)
            (width, height) = (image.shape[1], image.shape[0])
            tracking_position = template_position
            for step in xrange(1):
                image_clone = image.copy()
                tracking_position = self.tracking(image_clone, template_position, tracking_position)
                # iou = self.IOU(tracking_position, template_position)
                cv2.rectangle(image_clone , (tracking_position[0] , tracking_position[1]) ,
                             (tracking_position[2] , tracking_position[3]) , (255 , 0 , 0))
                cv2.imshow("tracking" , image_clone)
                cv2.waitKey()
            template_position = tracking_position

    def test_gym_env(self, checkpoint_file):
        self.load_model(checkpoint_file)
        image , template_position , tracking_position = self.env.reset()
        for i in xrange(5):
            template_image = image[template_position[1]:template_position[3] + template_position[1] ,
                             template_position[0]:template_position[2] + template_position[0] , :]
            tracking_image = image[tracking_position[1]:tracking_position[3] + tracking_position[1] ,
                             tracking_position[0]:tracking_position[2] + tracking_position[0] , :]
            template_image_resize = cv2.resize(template_image , (227 , 227))
            temp_image_resize = cv2.resize(tracking_image , (227 , 227))
            # image = np.concatenate([template_image_resize , temp_image_resize], axis=1)
            template_image_resize_feature = self.env.getFeatureByAlexNet(template_image_resize)
            tracking_image_resize_feature = self.env.getFeatureByAlexNet(temp_image_resize)

            action = self.get_action(np.concatenate(np.concatenate([template_image_resize_feature, tracking_image_resize_feature])))[0]

            state, reward, done, _ = self.env.step(action)
            print action, reward
            image = state[0]
            template_position = state[1]
            tracking_position = state[2]
            self.env.render("rgb_array")

    def IOU(self, Reframe , GTframe):
        x1 = Reframe[0];
        y1 = Reframe[1];
        width1 = Reframe[2] - Reframe[0];
        height1 = Reframe[3] - Reframe[1];

        x2 = GTframe[0];
        y2 = GTframe[1];
        width2 = GTframe[2] - GTframe[0];
        height2 = GTframe[3] - GTframe[1];

        endx = max(x1 + width1 , x2 + width2);
        startx = min(x1 , x2);
        width = width1 + width2 - (endx - startx);

        endy = max(y1 + height1 , y2 + height2);
        starty = min(y1 , y2);
        height = height1 + height2 - (endy - starty);

        if width <= 0 or height <= 0:
            ratio = 0
        else:
            Area = width * height;
            Area1 = width1 * height1;
            Area2 = width2 * height2;
            ratio = Area * 1. / (Area1 + Area2 - Area);
        # return IOU
        return ratio

    def tracking(self, image, template_position, tracking_position):
        '''

        :param image:
        :param template_position: left, top, right, bottom
        :param tracking_position:
        :return:
        '''

        template_image = image[template_position[1]:template_position[3] , template_position[0]:template_position[2], :]
        tracking_image = image[tracking_position[1]:tracking_position[3] , tracking_position[0]:tracking_position[2], :]
        template_image_resize = cv2.resize(template_image , (113 , 227))
        tracking_image_resize = cv2.resize(tracking_image , (114 , 227))
        image = np.concatenate([template_image_resize , tracking_image_resize] , axis=1)

        action = self.get_action(np.concatenate(self.env.getFeatureByAlexNet(image)))[0]
        print "tracking position:" + str(tracking_position)
        print "action:" + str(action)
        x_move = int(np.clip(action[0] , -1 , 1) * 0.3 * template_image.shape[1] + template_position[0])
        y_move = int(np.clip(action[1] , -1 , 1) * 0.3 * template_image.shape[0] + template_position[1])
        width_new = max(tracking_image.shape[1] * action[2], 50)
        heigh_new = max(tracking_image.shape[0] * action[3], 50)
        width_new = np.clip(width_new , 0 ,image.shape[1]).astype(int)
        heigh_new = np.clip(heigh_new , 0, image.shape[0]).astype(int)
        tracking_position = (x_move, y_move, x_move + width_new, y_move + heigh_new)
        return tracking_position

    def get_all_file_names(self , _path):
        file_list = os.listdir(_path)
        full_file_lists = []
        if file_list:
            for fn in file_list:
                full_file_name = os.path.join(_path , fn)
                full_file_lists.append(full_file_name)
        return full_file_lists

