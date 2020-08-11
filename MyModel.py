from keras.models import load_model
import DataGenerator
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau

tot_char = 3755
_epochs = 200
test_samples_per_class = 60

model_path = 'char_recognition.h5'


# initial_lr: 最开始的学习率
# charset_size 想要分类的汉字种类数目
# validationRate 验证集的比例
# pickedNumber: 每一个汉字训练集里取出多少图片(每一个总共有240个) 
# batch_size:每一个批的大小
def train(model, initial_lr = 0.001, img_size = (64, 64), charset_size=3755, validationRate = 0.3, pickedNumber = 240, batch_size = 128):
    dg = DataGenerator.DataGenerator()
    train_sample_count = dg.pick_small_dataset(char_number=charset_size, picked_number=pickedNumber, validation_rate=validationRate)
    train_generator, validation_generator, test_generator = dg.data_gen(batchSize=batch_size)

    _steps_per_epoch = charset_size * train_sample_count // batch_size
    _validation_steps = charset_size * (pickedNumber - train_sample_count) // batch_size

    model.compile(loss='categorical_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=initial_lr),
        metrics=['acc'])
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

    history = model.fit(
        train_generator,
        steps_per_epoch = _steps_per_epoch,
        epochs = _epochs,
        validation_data = validation_generator,
        validation_steps = _validation_steps,
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

def build(img_shape = (64, 64, 1), charset_size=3755):
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
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(charset_size, activation='softmax'))

    return model






# 加载model
model = load_model('char_recognition.h5')

dg = DataGenerator.DataGenerator()

# dg.pick_small_dataset(char_number=charset_size, picked_number=pickedNumber, validation_rate=validationRate)




# model = build(charset_size=50)
# test_generator, _steps = train(model, charset_size=50)

# test_loss, test_acc = model.evaluate(test_generator, steps=_steps)
# print('test acc: ', test_acc)