

class PMS_base(object):
    class_number = 26
    train_classify_frequency = 9
    test_frequency = 500
    test_epoche_size = 20
    train_file = "/home/aqrose/RL_toolbox/data/SUN360_panoramas_1024x512/train/train.txt"
    test_file = "/home/aqrose/RL_toolbox/data/SUN360_panoramas_1024x512/test/test.txt"
    history_number = 2  # image history number
    jobs = 4  # thread or process number
    max_iter_number = 4000  # 'control the max iteration number for trainning')
    paths_number = 10  # 'number of paths in each rollout')
    max_path_length = 3  # 'timesteps in each path')
    batch_size = 100  # 'batch size for trainning')
    max_kl = 0.01  # 'the largest kl distance  # \sigma in paper')
    gae_lambda = 1.0  # 'fix number')
    subsample_factor = 0.3  # 'ratio of the samples used in training process')
    GPU_fraction = 0.15
    cg_damping = 0.001  # 'conjugate gradient damping')
    discount = 0.99  # 'discount')
    cg_iters = 20  # 'iteration number in conjugate gradient')
    deviation = 0.1  # 'fixed')
    render = False  # 'whether to render image')
    train_flag = True  # 'true for train and False for test')
    iter_num_per_train = 1  # 'iteration number in each trainning process')
    checkpoint_file = ''  # 'checkpoint file path  # if empty then will load the latest one')
    save_model_times = 100  # 'iteration number to save model  # if 1  # then model would be saved in each iteration')
    upload_to_gym = False  # 'whether upload the result to gym')
    checkpoint_dir = 'checkpoint/'  # 'checkpoint save and load path  # for parallel  # it should be checkpoint_parallel')
    test_image_dir = "test_image_dir/"
    environment_name = 'ObjectTracker-v2'  # 'environment name')
    min_std = 1.2  # 'the smallest std')
    max_std = 2.4
    center_adv = True  # 'whether center advantage  # fixed')
    positive_adv = False  # 'whether positive advantage  # fixed')
    use_std_network = False  # 'whether use network to train std  # it is not supported  # fixed')
    std = 3.0  # 'if the std is set to constant  # then this value will be used')
    obs_shape = [4096]  # 'dimensions of observation')
    action_shape = 1  # 'dimensions of action')
    min_a = -3.14  # 'the smallest action value')
    max_a = 3.14  # 'the largest action value')
    entropy_beta = 0.1
    batch_size = int(subsample_factor * paths_number * max_path_length)