class DefaultConfigs(object):
    # actual parameters
    seed = 666
    weight_decay = 1e-4
    num_classes = 2
    resume = False
    pretrained = True
    model = 'resnet18'
    # hyper parameters
    gpus = "0"
    batch_size = 180
    norm_flag = True 

    tgt_best_model_name = '' # the model name that need to be tested

    # source data information
    src_data = 'msu'
    src_train_num_frames = 4
    src_data_label_path = "./data_label/" + src_data + "_data_label/"
    # target data information
    tgt_data = 'casia'
    tgt_train_num_frames = 2
    tgt_data_label_path = "./data_label/" + tgt_data + "_data_label/"
    # test data information
    test_num_frames = 5
    test_data = 'casia'
    test_label_path = "./data_label/" + test_data + "_data_label/"
    # paths information
    model_save_path = './' + src_data + '_checkpoint/'
    model_name = model_save_path + model + '/'
    best_model = model_name + 'best_model/'
    logs = './logs/'

config = DefaultConfigs()
