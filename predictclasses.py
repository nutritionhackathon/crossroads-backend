import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('./models/foodclasses.hdf5')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    rescale=1.0/255.0
)

classToName = {0:'Bread', 1:'Dairy product', 2: 'Dessert', 3: 'Egg', 4: 'Fried food', 5:'Meat', 6: 'Noodles-Pasta', 7: 'Rice', 8:'Seafood', 9:'Soup', 10: 'Vegetable-Fruit'}

image = tf.keras.preprocessing.image.load_img('bread_milk.webp', target_size=(300, 300))
image = tf.keras.preprocessing.image.img_to_array(image)

image = image.reshape(1, 300, 300, 3)
image = tf.keras.applications.vgg19.preprocess_input(image)
image = image / 255

print(model.predict(image))
print(classToName[np.argmax(model.predict(image))])