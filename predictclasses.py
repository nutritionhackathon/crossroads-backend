import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('./models/foodclasses.hdf5')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    rescale=1.0/255.0
)

classToName = {0:'Bread', 1:'Dairy product', 2: 'Dessert', 3: 'Egg', 4: 'Fried food', 5:'Meat', 6: 'Noodles-Pasta', 7: 'Rice', 8:'Seafood', 9:'Soup', 10: 'Vegetable-Fruit'}

def predictclasses(imageFile):
    image = tf.keras.preprocessing.image.load_img(imageFile, target_size=(300, 300))
    image = tf.keras.preprocessing.image.img_to_array(image)

    image = image.reshape(1, 300, 300, 3)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    image = image / 255

    arr = model.predict(image)
    arr = arr[0]
    h1 = 0; i1 = 0
    h2 = 0; i2 = 0
    h3 = 0; i3 = 0
    for i in range(11):
        if arr[i] > h1:
            h3 = h2; h2 = h1; h1 = arr[i]; i1 = i
        elif arr[i] > h2:
            h3 = h2; h2 = arr[i]; i2 = i
        elif arr[i] > h3:
            h3 = arr[i]; i3 = i
    return [classToName[i1], classToName[i2], classToName[i3]]
