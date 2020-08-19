from keras.models import load_model
import DataGenerator
from keras import layers
from keras import models
from keras.models import Model
from keras import optimizers
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

tot_char = 3755
_epochs = 200
test_samples_per_class = 60

model_path = 'char_recognition.h5'


# initial_lr: 最开始的学习率
# charset_size 想要分类的汉字种类数目
# validationRate 验证集的比例
# pickedNumber: 每一个汉字训练集里取出多少图片(每一个总共有240个) 
# batch_size:每一个批的大小
def train(model, initial_lr = 0.001, img_size = (64, 64), charset_size=3755, validationRate = 0.3, pickedNumber = 240, batch_size = 2024):
    dg = DataGenerator.DataGenerator()
    # train_sample_count = dg.pick_small_dataset(char_number=charset_size, picked_number=pickedNumber, validation_rate=validationRate)
    train_generator, test_generator = dg.data_gen(batchSize=batch_size)

    train_sample_count = charset_size * pickedNumber

    _steps_per_epoch = charset_size * train_sample_count // batch_size
    # _validation_steps = charset_size * (pickedNumber - train_sample_count) // batch_size

    model.compile(loss='categorical_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=initial_lr),
        metrics=['acc'])
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

    history = model.fit(
        train_generator,
        steps_per_epoch = _steps_per_epoch,
        epochs = _epochs,
        # validation_split= validationRate,
        # validation_steps = _validation_steps,
        callbacks=[reduce_lr]
    )
    model.save(model_path)

    display(history)

    return test_generator, charset_size * test_samples_per_class // batch_size # 这个参数是model.evaluate()里 steps参数的值
  
# 训练后展示准确率以及损失变化情况
def display(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    
# 注意这里的charset_size一定要与 train里的charset_size相同

def build(model_type = None, img_shape=(64, 64, 3), charset_size=3755):
    if model_type == 'resnet50':
        # base_model = ResNet50(include_top=False, weights=None)
        # base_model = models.Sequential(input = base_model.input, output=base_model.get_layer(index=175).output)

        # base_model.add(layers.Dropout(0.3))
        # base_model.add(layers.Dense(512, activation='relu'))
        # base_model.add(layers.Dense(charset_size, activation='softmax'))
        # return base_model

        base_model = ResNet50(
            weights = None,
            classes = charset_size,
            input_shape=img_shape
        )
        return base_model
    elif model_type == 'vgg16':
        base_model = VGG16(include_top=True, weights=None)
        base_model = models.Sequential(input = base_model.input, output=base_model.get_layer(index=21).output)
        
        base_model.add(layers.Dropout(0.3))
        base_model.add(layers.Dense(512, activation='relu'))
        base_model.add(layers.Dense(charset_size, activation='softmax'))
        return base_model

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(charset_size, activation='softmax'))

    return model



# char_num = 15
# model = build('vgg16', charset_size=char_num)

# testgen, _step = train(model, charset_size=char_num, batch_size=28, pickedNumber=100)

# test_loss, test_acc = model.evaluate(testgen, steps = _step)

# print('test acc: ', test_acc)
# print('test loss: ', test_loss)

# 加载model
# model = load_model('char_recognition.h5')

# dg = DataGenerator.DataGenerator()

# dg.pick_small_dataset(char_number=charset_size, picked_number=pickedNumber, validation_rate=validationRate)




model = build(model_type='resnet50')
test_generator, _steps = train(model)

test_loss, test_acc = model.evaluate(test_generator, steps=_steps)
print('test acc: ', test_acc)