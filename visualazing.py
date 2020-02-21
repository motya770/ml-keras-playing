import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

print(keras.__version__)


model = load_model('/Users/kudelin/Desktop/work/projects/ml-keras-playing/cats_and_dogs_small_2.h5')
model.summary()  # As a reminder.

img_path = '/Users/kudelin/Downloads/catanddogs/cats_and_dogs_small/test/cats/cat.1700.jpg'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor /= 255.

# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()