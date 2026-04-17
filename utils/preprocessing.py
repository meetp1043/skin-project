import tensorflow as tf

def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image
