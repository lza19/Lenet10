from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# โหลดโมเดลจากไฟล์
model = load_model(r'C:\CNNet\lenet5\lenet5_mnist.h5')

# โหลดรูปภาพ (เช่น 'path_to_image.jpg')
img_path = r'C:\InceptionNet\mnist_png\test\9\7.png'
img = image.load_img(img_path, target_size=(32, 32), color_mode='grayscale')

# แปลงรูปภาพเป็นอาร์เรย์
img_array = image.img_to_array(img)

# ปรับขนาดให้เป็น (1, 32, 32, 1) เนื่องจากโมเดลคาดหวัง input ขนาดนี้
img_array = np.expand_dims(img_array, axis=0)

# ทำการปรับค่าให้เหมาะสม เช่น rescale ถ้าใช้ ImageDataGenerator
img_array /= 255.0

# ทำนายผล
prediction = model.predict(img_array)

# แสดงผลลัพธ์ (ค่าความน่าจะเป็น)
print("Prediction:", prediction)

# ถ้าต้องการให้แสดงผลเป็นคลาสที่คาดการณ์ได้
predicted_class = np.argmax(prediction, axis=1)
print("Predicted Class:", predicted_class)
