import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

train_data_generator = image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_data_generator = image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_generator.flow_from_directory(
    'fish_dataset/train',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_data_generator.flow_from_directory(
    'fish_dataset/validation',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
output = base_model.output
output = Flatten()(output)
output = Dense(1024, activation='relu')(output)
predictions = Dense(train_generator.num_classes, activation='softmax')(output)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

model.save('trained_model.h5')

model = load_model('trained_model.h5')


def identify_fish_specie(data):
    img = image.load_img(data, target_size=(299, 299)) if type(data) is str else cv2.resize(data, (299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    predicted = model.predict(img)
    predicted_class = np.argmax(predicted, axis=1)
    class_labels = {v: k for k, v in train_generator.class_indices.items()}

    return class_labels[predicted_class[0]]


print(f'Identified fish specie: {identify_fish_specie("fish_dataset/test/Gold Fish/Gold Fish 1.jpg")}')
print(f'Identified fish specie: {identify_fish_specie("fish_dataset/test/Tilapia/Picture4.jpg")}')
print(f'Identified fish specie: {identify_fish_specie("fish_dataset/test/Catfish/0efe7c4b-76e6-4461-9db8-29fda1c30d3e-710mm.jpg")}')
print(f'Identified fish specie: {identify_fish_specie("fish_dataset/test/Janitor Fish/Janitor Fish 1.jpg")}')

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    if not ret:
        break

    cv2.putText(frame, f'Identified fish specie: {identify_fish_specie(frame)}', (8, 28), cv2.QT_FONT_NORMAL, .8, (255, 132, 0), 1, cv2.LINE_AA)
    cv2.imshow('Press "ESC" to exit', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
