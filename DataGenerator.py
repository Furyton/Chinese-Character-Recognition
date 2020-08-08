import json
import os
import shutil
import numpy as np

# from keras.preprocessing.image import ImageDataGenerator

class_list_dir = "C:/Users/Furyton/Documents/Machine Learning/Original_dataset/CASIA-Classes.txt"
original_data_dir = "C:/Users/Furyton/Documents/Machine Learning/Original_dataset"
base_data_dir = "C:/Users/Furyton/Documents/Machine Learning/small_dataset"


class DataGenerator:
    def __init__(self):
        self.ch_num_dict = {} # a dict from character to its order
        self.num_ch_dict = {} # a dict from order to its charactor

    # load file "CASIA-Classes.txt"
    def load_class_list(self):
        content = None
        with open(class_list_dir, 'r', encoding='utf-8') as file:
            content = file.read()

        content = content.replace("'", '"')
        content = content[content.find('{'):]
        
        self.ch_num_dict = json.loads(content)

        self.num_ch_dict = {val:key for key, val in self.ch_num_dict.items()}
    
    # 从总的数据集中选出一个较小的训练集, 从训练集分出验证集, 测试集保持不变
    def pick_small_dataset(self, char_number = 3755,  picked_number = 240, validation_rate = 0.2):
        if self.ch_num_dict is None:
            self.load_class_list()
        
        os.mkdir(base_data_dir)

        train_dir = os.path.join(base_data_dir, 'train')
        os.mkdir(train_dir)
        
        validation_dir = os.path.join(base_data_dir, 'validation')
        os.mkdir(validation_dir)

        test_dir = os.path.join(base_data_dir, 'test')
        os.mkdir(test_dir)

        # [train and validation]
        original_train = os.path.join(original_data_dir, 'train')

        validation_count = (int)(picked_number * validation_rate)
        train_count = picked_number - validation_count
        
        ClassFileList = [name for name in os.listdir(original_train)if os.path.isdir(os.path.join(original_train, name))][:char_number]
        
        for fileName in ClassFileList:
            org_path = os.path.join(original_train, fileName)

            order = self.ch_num_dict[fileName]
            new_train = os.path.join(train_dir, str(order))
            os.mkdir(new_train)

            new_vali = os.path.join(validation_dir, str(order))
            os.mkdir(new_vali)

            PNGFileList = os.listdir(org_path)

            np.random.shuffle(PNGFileList)

            cnt = 0

            for pngName in PNGFileList[:train_count]:
                src = os.path.join(org_path, pngName)
                dst = os.path.join(new_train, str(order) + '.' + str(cnt) + '.png')
                shutil.copyfile(src, dst)

                cnt += 1
            for pngName in PNGFileList[train_count:picked_number]:
                src = os.path.join(org_path, pngName)
                dst = os.path.join(new_vali, str(order) + '.' + str(cnt) + '.png')
                shutil.copyfile(src, dst)

                cnt += 1

        # [test]

        original_test = os.path.join(original_data_dir, 'test')

        for fileName in ClassFileList:
            org_path = os.path.join(original_test, fileName)

            order = self.ch_num_dict[fileName]
            new_test = os.path.join(test_dir, str(order))
            os.mkdir(new_test)

            PNGFileList = os.listdir(org_path)

            cnt = 0

            for pngName in PNGFileList:
                src = os.path.join(org_path, pngName)
                dst = os.path.join(new_test, str(order) + '.' + str(cnt) + '.png')
                shutil.copyfile(src, dst)

                cnt += 1

test = DataGenerator()

test.load_class_list()
# test.pick_small_dataset(char_number=2, picked_number=2, validation_rate=0.5)