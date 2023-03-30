# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Загружаем данные MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Просмотр первых 10 изображений
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
plt.show()

# Нормализация данных
x_train = x_train / 255.0
x_test = x_test / 255.0

# Преобразуем метки классов в one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Инициализируем модель
model = Sequential()

# Добавляем сверточный слой с 32 фильтрами и ядром 3x3
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Добавляем еще один сверточный слой с 64 фильтрами и ядром 3x3
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Добавляем слой подвыборки (MaxPooling) с ядром 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Преобразуем выходные данные из слоя подвыборки в одномерный массив
model.add(Flatten())

# Добавляем полносвязный слой с 128 нейронами
model.add(Dense(128, activation='relu'))

# Добавляем выходной слой с 10 нейронами и функцией активации softmax
model.add(Dense(10, activation='softmax'))

# Компилируем модель с функцией потерь categorical_crossentropy и оптимизатором Adam
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучаем модель на данных MNIST
model.fit(np.expand_dims(x_train, axis=-1), y_train, batch_size=128, epochs=5, verbose=1)

# Оцениваем качество модели на тестовой выборке
test_loss, test_acc = model.evaluate(np.expand_dims(x_test, axis=-1), y_test)
print('Кэф точности:', test_acc)

# Визуализируем несколько случайных изображений из тестовой выборки и выведем предсказание модели для каждого изображения
indices = np.random.choice(len(x_test), 5)

for index in indices:
    # Вывод изображения
    plt.imshow(x_test[index], cmap='gray')
    plt.show()

    # Предсказание модели
    prediction = model.predict(np.expand_dims(x_test[index], axis=0))
    predicted_label = np.argmax(prediction)
    print('Predicted label:', predicted_label)

