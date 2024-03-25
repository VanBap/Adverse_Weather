import tensorflow as tf
import pathlib

# Load model
model=tf.keras.models.load_model('F:/saved_model/ResNet50_Weather_epoch20.h5')

# Đường dẫn tới thư mục chứa ảnh
data_dir = "F:/CODE_PYCHARM/KhoaLuan/saved_model/DataSet/dataset2"

# Lấy danh sách các đường dẫn tới các tệp ảnh trong thư mục
data_root = pathlib.Path(data_dir)
all_image_paths = list(data_root.glob('*/*'))

# Hàm lambda trả về một dataset TensorFlow từ các đường dẫn ảnh
def representative_dataset_gen():
    for img_path in all_image_paths:
        # Đọc ảnh từ đường dẫn
        img = tf.io.read_file(str(img_path))
        # Xử lý ảnh (ví dụ: giảm kích thước xuống 224x224, chuẩn hóa,...)

        # Chuyển đổi kiểu dữ liệu và định dạng tensor
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (224, 224))  # Giả sử kích thước đầu vào của model là (224, 224, 3)
        img = tf.expand_dims(img, 0)  # Thêm một chiều cho batch
        yield [img]


# Define input shape for the model
input_shape = (1, 224, 224, 3)

# Convert the model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Cài đặt kích thước đầu vào và đầu ra của model tflite
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Set input shape for the TFLite model
converter.representative_dataset = representative_dataset_gen

# Convert model và lưu vào file .tflite
tflite_model = converter.convert()
with open('F:\CODE_PYCHARM\KhoaLuan\saved_model/ResNet50_Weather_epoch20.tflite', 'wb') as f:
    f.write(tflite_model)

