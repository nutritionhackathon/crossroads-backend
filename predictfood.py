import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('./models/foodidentification.hdf5')


classes = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio',
           'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding',
           'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad',
           'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
           'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich',
           'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings',
           'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
           'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
           'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza',
           'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque',
           'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings',
           'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
           'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
           'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi',
           'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']



def predictfood(filename):
    img_ = image.load_img(filename, target_size=(228, 228))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)

    index = np.argmax(prediction)

    f = classes[index]
    if f == 'apple_pie' or f == 'baklava' or f == 'beignets' or f == 'bread_pudding' or f == 'cannoli' or f == 'carrot_cake' or f == 'cheescake' or f == 'chocolate_cake' or f == 'chocolate_mousse' or f == 'churros' or f == 'creme_brulee' or f == 'cup_cakes' or f == 'donuts' or f == 'ice_cream' or f == 'macarons' or f == 'panna_cotta' or f == 'red_velvet_cake' or f == 'strawberry_shortcake' or f == 'tiramisu':
        return [f, 'salad', 'fruit', 'yogurt']
    elif f == 'beet_salad' or f == 'caesar_salad' or f == 'caprese_salad' or f == 'greek_salad':
        return [f, 'nuts', 'seeds', 'avocado']
    elif f == 'baby_back_ribs' or f == 'beef_carpaccio' or f == 'beef_tartare' or f == 'pork_chop' or f == 'prime_rib' or f == 'steak':
        return [f, 'vegetables', 'grains', 'fruit']
    elif f == 'chicken_quesadilla' or f == 'chicken_wings' or f == 'club_sandwich' or f == 'croque_madame' or f == 'fish_and_chips' or f == 'french_fries' or f == 'fried_calamari' or f == 'grilled_chesse_sandwich' or f == 'macaroni_and_cheese' or f == 'nachos' or f == 'onion_rings' or f == 'pizza':
        return [f, 'vegetables', 'fruit', 'grains']
    return [f, 'salad', 'fruit', 'yogurt']