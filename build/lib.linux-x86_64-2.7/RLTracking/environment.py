from RLToolbox.environment.gym_environment import Environment
import cv2
import numpy as np

class EnvironmentTracking(Environment):
    def __init__(self, env, pms, type="origin"):
        super(EnvironmentTracking, self).__init__(env, pms, type)

    def step(self , action , **kwargs):
        self._observation , reward , done , info = self.env.step(action)
        return self.observation , reward , done , info

    def render(self , mode="human" , close=False):
        if mode == "human":
            return self.env.render(mode)
        elif mode == "rgb_array":
            tuple = self.env.render('rgb_array')
            template_image = tuple[0]
            temp_image = tuple[1]
            template_image_resize = cv2.resize(template_image , (self.pms.obs_shape[1]/2 , self.pms.obs_shape[0]))
            temp_image_resize = cv2.resize(temp_image , (self.pms.obs_shape[1]/2 , self.pms.obs_shape[0]))
            image = np.concatenate([template_image_resize , temp_image_resize], axis=1)
            # cv2.imshow("source_image", image)
            # cv2.waitKey()
            return image