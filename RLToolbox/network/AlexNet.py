from RLToolbox.network.network import Network
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from RLToolbox.network.alexNet.bvlc_alexnet import AlexNet as AlexNetKaffe

class AlexNet(Network):
    def __init__(self, scope, pms):
        super(AlexNet, self).__init__(scope, pms)
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                tf.float32 , shape=[None] + [pms.obs_shape[0] , pms.obs_shape[1] , pms.obs_shape[2]] ,
                name="%s_obs" % scope)
            self.obs_target = obs_target = tf.placeholder(
                tf.float32 , shape=[None] + pms.obs_shape , name="%s_obs_target" % scope)
            self.action_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] , name="%s_action" % scope)
            self.advant = tf.placeholder(tf.float32 , shape=[None] , name="%s_advant" % scope)
            self.old_dist_means_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] ,
                                                   name="%s_oldaction_dist_means" % scope)
            self.old_dist_logstds_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] ,
                                                     name="%s_oldaction_dist_logstds" % scope)

            self.network = AlexNetKaffe({'data': self.obs})
            self.action_dist_means_n = self.network.layers['train_fc11']
            self.N = tf.shape(obs)[0]
            self.action_dist_logstd_param = tf.Variable(
                (.01 * np.random.randn(1 , pms.action_shape)).astype(np.float32) , name="train_%s_policy_logstd" % scope)
            # self.action_dist_logstd_param = tf.maximum(self.action_dist_logstd_param, np.log(pms.min_std))
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param ,
                                                 tf.pack((tf.shape(self.action_dist_means_n)[0] , 1)))
            start_name = "%s_shared/%s" %(scope, "train")
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(start_name)]

    def asyc_parameters(self, session=None):
        self.network.load("bvlc_alexnet_model.npy", session, scope="%s_shared" % self.scope, ignore_missing=True)