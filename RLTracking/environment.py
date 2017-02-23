from RLToolbox.environment.gym_environment import Environment
import cv2
import numpy as np
import tensorflow as tf
from RLToolbox.network.alexNet.bvlc_alexnet import AlexNet
class EnvironmentTracking(Environment):
    def __init__(self, env, pms, type="origin", session=None):
        super(EnvironmentTracking, self).__init__(env, pms, type)
        self.alexNetSess = session
        self.initAlexNet()

    def initAlexNet(self):
        self.my_input_data = tf.placeholder(dtype='float32' , shape=[None , 227 , 227 , 3] , name="alex_input_data")
        alexNet = AlexNet({'data': self.my_input_data})
        alexNet.load("bvlc_alexnet_model.npy" , self.alexNetSess, scope=None, ignore_missing=True)
        self.fc7 = alexNet.layers['fc7']
        print self.fc7

    # Forward pass

    def getFeatureByAlexNet(self, image):
        pp = cv2.resize(image, (227 , 227))
        pp = np.asarray(pp, dtype=np.float32)
        pp = pp.reshape((1, pp.shape[1], pp.shape[0], 3))

        return self.alexNetSess.run(self.fc7, feed_dict={self.my_input_data: pp})

    def step(self , action , **kwargs):
        self._observation , reward , done , info = self.env.step(action)
        return self.observation , reward , done , info

    def render(self , mode="human" , close=False):
        if mode == "human":
            return self.env.render(mode)
        elif mode == "rgb_array":
            tuple = self.env.render()
            source_image = tuple[0]
            template_position = tuple[1]
            tracking_position = tuple[2]
            template_image = source_image[template_position[1]:template_position[3] + template_position[1] ,
                             template_position[0]:template_position[2] + template_position[0] , :]
            tracking_image = source_image[tracking_position[1]:tracking_position[3] + tracking_position[1] ,
                             tracking_position[0]:tracking_position[2] + tracking_position[0] , :]
            # image = np.concatenate([template_image_resize , temp_image_resize], axis=1)
            template_image_resize_feature = self.getFeatureByAlexNet(template_image)
            tracking_image_resize_feature = self.getFeatureByAlexNet(tracking_image)
            env_image_clone = source_image.copy()
            template_left_top = (template_position[0] , template_position[1])
            template_right_bottom = (template_position[0] + template_position[2] ,
                                     template_position[1] + template_position[3])
            tracking_left_top = (tracking_position[0] , tracking_position[1])
            tracking_right_bottom = (tracking_position[0] + tracking_position[2] ,
                                     tracking_position[1] + tracking_position[3])
            cv2.rectangle(env_image_clone , template_left_top , template_right_bottom , (255 , 0 , 0))
            cv2.rectangle(env_image_clone , tracking_left_top , tracking_right_bottom , (0 , 0 , 255))
            # cv2.imshow("show test" , env_image_clone)
            # cv2.waitKey()
            return np.concatenate(np.concatenate([template_image_resize_feature, tracking_image_resize_feature]))