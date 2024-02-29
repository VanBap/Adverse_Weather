import tensorflow as tf

# Load model
model=tf.keras.models.load_model('F:/CODE_PYCHARM/KhoaLuan/saved_model/ResNet50_Weather.h5')

# Khởi tạo 1 bộ converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Thực hiện convert

tflite_model = converter.convert()

# Write vao file
open("F:/CODE_PYCHARM/KhoaLuan/saved_model/model.tflite", "wb").write(tflite_model)