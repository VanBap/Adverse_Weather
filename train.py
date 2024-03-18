from google.colab import drive
drive.mount('/content/gdrive')

from keras.applications import MobileNet
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# MobileNet is designed to work with images of dim 224,224
img_rows,img_cols = 224,224

MobileNet = MobileNet(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,3))

for layer in MobileNet.layers:
    layer.trainable = True

# Let's print our layers
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i),layer.__class__.__name__,layer.trainable)
