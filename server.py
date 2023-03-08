# 服务端程序：
# 导入flask库
from flask import Flask, request, jsonify
import joblib
from PIL import Image
import numpy as np

# 加载手写数字图片识别模型
clf_ld_mod = joblib.load(r'./model/svm_model.joblib')

# 创建flask应用对象
app = Flask(__name__)


# 定义路由函数，处理客户端提交的音频数据，并返回识别结果
@app.route('/prediction', methods=['POST'])
def predict():
    # 获取音频数据
    file = request.files['file']
    # print(file)
    image = np.array(Image.open(file).convert('L'))
    # 调用模型，得到文字内容
    prediction = clf_ld_mod.predict(image.reshape(1, -1))

    # 将结果返回给客户端
    result = {'prediction': int(prediction[0])}
    return jsonify(result)


# 运行flask应用，监听5000端口
app.run(port=5000)
