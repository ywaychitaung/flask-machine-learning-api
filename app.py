from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import tensorflow as tf
import numpy as np
import os
import io
from keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model from the h5 file
# 从h5文件中加载模型
model = tf.keras.models.load_model('models/image_classification/image_classification_model.h5')

# Define class names
# 定义类别名称
class_names = ['apple', 'banana', 'mixed', 'orange']

def preprocess_image(file_content):
    # Load and preprocess the image
    # 载入和预处理图像
    img = image.load_img(file_content, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def get_prediction(img_array):
    # Make predictions
    # 进行预测
    predictions = model.predict(img_array)

    # Get the predicted class index
    # 获取预测的类别索引
    predicted_class = int(np.argmax(predictions))

    # Map the class index to the class label
    # 将类别索引映射到类别标签
    predicted_class_label = class_names[predicted_class]

    # Apply softmax to convert raw predictions to probabilities
    # 应用softmax将原始预测转换为概率
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=-1)

    return predicted_class_label, probabilities[0]

@app.route('/image-classification', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file part', 400

    # Get the image from the request
    # 从请求中获取图像
    uploaded_file = request.files['image']

    if uploaded_file.filename == '':
        return 'No selected file', 400

    # Read the file content into a BytesIO object
    # 将文件内容读取到BytesIO对象中
    file_content = io.BytesIO(uploaded_file.read())

    # Preprocess the image
    # 预处理图像
    img_array = preprocess_image(file_content)

    # Get prediction
    # 进行预测
    predicted_class_label, probabilities = get_prediction(img_array)

    print(predicted_class_label, probabilities)

    # Return the predicted class label and probability scores as a response
    # 作为响应返回预测的类别标签和概率分数
    return jsonify({
        'prediction': predicted_class_label, 
        'probabilities': 
            {
                class_name: prob.item() for class_name, prob in zip(class_names, probabilities)
            }
    })

if __name__ == "__main__":
     app.run(host='0.0.0.0', port=5000)