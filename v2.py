import time
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# преобразование размерности изображений и нормализация пикселей
X_train = np.expand_dims(X_train, axis=-1) / 255.0
X_test = np.expand_dims(X_test, axis=-1) / 255.0

# разделение данных на обучающую и тестовую выборки
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# создание модели
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

# компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# обучение модели
start_time = time.time()  # сохраняем время начала обучения
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))
end_time = time.time()  # сохраняем время окончания обучения

# вывод информации о времени обучения и точности
print(f'Время тренировки: {end_time - start_time:.2f} seconds')
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Кэф точности: {test_acc:.3f}')

# вывод предсказаний для нескольких изображений из тестовой выборки
n_images = 10
for i in range(n_images):
    image_index = np.random.choice(len(X_test))
    image = X_test[image_index]
    true_label = y_test[image_index]
    predicted_probs = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(predicted_probs, axis=-1)[0]
    is_correct = predicted_label == true_label
    print(f'Изображение {i+1}: {"Корректно" if is_correct else "Некорректно"} - Предсказано: {predicted_label}, Истинное значение: {true_label}')


# выбираем случайное изображение из тестовой выборки
image_index = np.random.choice(len(X_test))
image = X_test[image_index]
true_label = y_test[image_index]

# получаем предсказания модели для изображения
predicted_probs = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(predicted_probs, axis=-1)[0]

# выводим изображение и предсказание модели
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Предсказанное значение: {predicted_label}, Истинное значение: {true_label}")
plt.show()