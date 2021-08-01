import os
import json

# base_dir = '/home/jiayunpei/data/anti_spoofing_processed_data'
base_dir = '$root/processed_data/' # change to your own path

def replay_generate_label():
    data_dir = base_dir + '/replay_processed_data_256/'
    label_save_dir = './replay_data_label/'
    print('\n === replay dataset === \n')
    if not os.path.exists(label_save_dir):
        os.mkdir(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')

    replayattack_list = os.listdir(data_dir)
    print(replayattack_list)
    for dir_1 in replayattack_list:
        final_json = []
        count = 0
        replayattach_dir = os.path.join(data_dir, dir_1)
        type_list = os.listdir(replayattach_dir)
        for video_type in type_list:
            video_dir = ''
            video_list = []
            if(video_type == 'real'):
                video_dir = os.path.join(replayattach_dir, video_type)
                video_list = os.listdir(video_dir)
                video_list.sort()
                for photo in video_list:
                    file_path = os.path.join(video_dir, photo)
                    frame_num = photo.split('.')[-2]
                    video_num = photo.split('.')[-3]
                    if (video_type == 'real'):
                        label = 1
                    else:
                        label = 0
                    dict = {}
                    dict['photo_path'] = file_path
                    dict['photo_label'] = label
                    final_json.append(dict)
                    count = count + 1
                    if (count % 10000 == 0):
                        print(count)
            else:
                new_dir = os.path.join(replayattach_dir, video_type)
                new_type_list = os.listdir(new_dir)
                for new_type in new_type_list:
                    video_dir = os.path.join(new_dir, new_type)
                    video_list = os.listdir(video_dir)
                    video_list.sort()
                    for photo in video_list:
                        file_path = os.path.join(video_dir, photo)
                        if (video_type == 'real'):
                            label = 1
                        else:
                            label = 0
                        dict = {}
                        dict['photo_path'] = file_path
                        dict['photo_label'] = label
                        final_json.append(dict)
                        count = count + 1
                        if (count % 10000 == 0):
                            print(count)
        print("--------", dir_1, ": ", count, " --------")
        if(dir_1 == 'replayattack-train'):
            json.dump(final_json, f_train, indent=4)
            f_train.close()
        elif(dir_1 == 'replayattack-devel'):
            json.dump(final_json, f_valid, indent=4)
            f_valid.close()
        else:
            json.dump(final_json, f_test, indent=4)
            f_test.close()

def msu_generate_label():
    data_dir = base_dir + '/msu_processed_data_256/'
    label_save_dir = './msu_data_label/'
    print('\n === msu dataset === \n')
    test_list = []
    for line in open('/data/share/jiayunpei/original_data/MSU-MFSD/test_sub_list.txt', 'r'):
        test_list.append(line[0:2])
    train_list = []
    for line in open('/data/share/jiayunpei/original_data/MSU-MFSD/train_sub_list.txt', 'r'):
        train_list.append(line[0:2])
    if not os.path.exists(label_save_dir):
        os.mkdir(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')

    type_list = os.listdir(data_dir)

    train_final_json =[]
    train_video_num = 0
    test_video_num = 0
    test_final_json = []
    for dir_1 in type_list:  # attack / real
        count = 0
        type_path = os.path.join(data_dir, dir_1)
        number_list = os.listdir(type_path)
        for dir_2 in number_list:
            number_path = os.path.join(type_path, dir_2)
            photo_list = os.listdir(number_path)
            photo_list.sort()
            number = dir_2.split('_')[0]
            if (number in train_list):
                train_video_num += 1
            else:
                test_video_num += 1
            for photo in photo_list:
                photo_path = os.path.join(number_path, photo)
                video_num = photo.split('.')[-3]
                if(dir_1 == 'real'):
                    label = 1
                else:
                    label = 0
                dict = {}
                dict['photo_path'] = photo_path
                dict['photo_label'] = label
                if(video_num in train_list):
                    train_final_json.append(dict)
                else:
                    test_final_json.append(dict)
                count = count + 1
                if (count % 10000 == 0):
                    print(count)
    print('-----train video num: ', train_video_num,'-----')
    print('-----test video num: ', test_video_num, '-----')
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()

def casia_generate_label():
    data_dir = base_dir + '/casia_processed_data_256/'
    label_save_dir = './casia_data_label/'
    print('\n === casia dataset === \n')
    if not os.path.exists(label_save_dir):
        os.mkdir(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')

    release_list = os.listdir(data_dir)
    for dir_1 in release_list:
        final_json = []
        count = 0
        release_path = os.path.join(data_dir, dir_1)
        number_list = os.listdir(release_path)
        for dir_2 in number_list:
            number_path = os.path.join(release_path, dir_2)
            photo_list = os.listdir(number_path)
            photo_list.sort()
            for photo in photo_list:
                photo_path = os.path.join(number_path, photo)
                frame_num = photo.split('.')[-2]
                video_num = photo.split('.')[-3]
                if (video_num == '1' or video_num == '2' or video_num == 'HR_1'):
                    label = 1
                else:
                    label = 0 
                dict = {}
                dict['photo_path'] = photo_path
                dict['photo_label'] = label
                final_json.append(dict)
                count = count + 1
                if (count % 10000 == 0):
                    print(count)
        print("--------", dir_1, ": ", count, " --------")
        if(dir_1 == 'train_release'):
            json.dump(final_json, f_train, indent=4)
            f_train.close()
        else:
            json.dump(final_json, f_test, indent=4)
            f_test.close()

if __name__=="__main__":
    replay_generate_label()
    msu_generate_label()
    casia_generate_label()