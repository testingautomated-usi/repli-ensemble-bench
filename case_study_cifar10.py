from typing import Dict

import uncertainty_wizard as uwiz

import gpu_db_recorder
from gpu_db_recorder import Event


class MultiGpuContext(uwiz.models.ensemble_utils.DeviceAllocatorContextManager):

    @classmethod
    def file_path(cls) -> str:
        return "temp"

    @classmethod
    def run_on_cpu(cls) -> bool:
        return False

    @classmethod
    def virtual_devices_per_gpu(cls) -> Dict[int, int]:
        return {
            0: 3,
            1: 3
        }

    @classmethod
    def gpu_memory_limit(cls) -> int:
        return 1024


def train_model(model_id):
    import tensorflow as tf

    gpu_db_recorder.dump(Event("Start model creation", model=model_id))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                                     input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    history = model.fit(x_train, y_train, batch_size=32, epochs=100)

    gpu_db_recorder.dump(Event("End model creation", model=model_id))

    return model, history.history
