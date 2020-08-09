import json
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

class_list_dir = "C:/Users/Furyton/Documents/Machine Learning/Original_dataset/CASIA-Classes.txt"
original_data_dir = "C:/Users/Furyton/Documents/Machine Learning/Original_dataset"
base_data_dir = "C:/Users/Furyton/Documents/Machine Learning/small_dataset"


class DataGenerator:
    def __init__(self):
        self.ch_num_dict = {} # a dict from character to its order
        self.num_ch_dict = {} # a dict from order to its charactor

        self.cur_charset = None
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
    def pick_small_dataset(self, char_number = 3755,  picked_number = 240, validation_rate = 0.2, ignoreExistence = False):
        # char_number 想要训练的汉字种类大小
        # picked_number 每个种类的汉字选出样本图片的数目
        # validation_rate 验证集占训练集的比例
        # ignoreExistence 当已经存在  base_data_dir 时是否继续(若存在则删除)
        if self.ch_num_dict is None:
            self.load_class_list()
        
        if os.path.exists(base_data_dir):
            if not ignoreExistence:
                print("文件夹已存在,操作取消")
                return
            else:
                shutil.rmtree(base_data_dir)

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
        self.cur_charset = ClassFileList

        for fileName in ClassFileList:
            org_path = os.path.join(original_train, fileName)

            new_train = os.path.join(train_dir, fileName)
            os.mkdir(new_train)

            new_vali = os.path.join(validation_dir, fileName)
            os.mkdir(new_vali)

            PNGFileList = os.listdir(org_path)

            np.random.shuffle(PNGFileList)

            cnt = 0

            for pngName in PNGFileList[:train_count]:
                src = os.path.join(org_path, pngName)
                dst = os.path.join(new_train, fileName + '.' + str(cnt) + '.png')
                shutil.copyfile(src, dst)

                cnt += 1
            for pngName in PNGFileList[train_count:picked_number]:
                src = os.path.join(org_path, pngName)
                dst = os.path.join(new_vali, fileName + '.' + str(cnt) + '.png')
                shutil.copyfile(src, dst)

                cnt += 1

        # [test]

        original_test = os.path.join(original_data_dir, 'test')

        for fileName in ClassFileList:
            org_path = os.path.join(original_test, fileName)

            new_test = os.path.join(test_dir, fileName)
            os.mkdir(new_test)

            PNGFileList = os.listdir(org_path)

            cnt = 0

            for pngName in PNGFileList:
                src = os.path.join(org_path, pngName)
                dst = os.path.join(new_test, fileName + '.' + str(cnt) + '.png')
                shutil.copyfile(src, dst)

                cnt += 1

    def data_gen(self, base_dir = base_data_dir, batchSize = 512, targetSize = (64, 64)):
        if not os.path.exists(base_dir):
            print("数据集所在目录不存在")
            return
        
        train_dir = os.path.join(base_data_dir, 'train')
        
        validation_dir = os.path.join(base_data_dir, 'validation')

        test_dir = os.path.join(base_data_dir, 'test')


        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=targetSize,
            batch_size=batchSize,
            class_mode='categorical',
            color_mode='grayscale',
            classes=self.cur_charset
        )
        validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=targetSize,
            batch_size=batchSize,
            class_mode='categorical',
            color_mode='grayscale',
            classes=self.cur_charset
        )
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=targetSize,
            batch_size=batchSize,
            class_mode='categorical',
            color_mode='grayscale',
            classes=self.cur_charset
        )

        return train_generator, validation_generator, test_generator

# test = DataGenerator()

# test.load_class_list()
# test.pick_small_dataset(char_number=20, picked_number=10, validation_rate=0, ignoreExistence=False)

# gen = test.data_gen()

# for data_batch, label_batch in gen:
#     plt.imshow(image.array_to_img(data_batch[0]))
#     print(label_batch[0])
#     break
# plt.show()