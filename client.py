# 导入requests库和PIL库
import requests
import argparse
# from PIL import Image
import sys


def main(_args):
    parser = argparse.ArgumentParser(description='Handwritten digital picture text recognition client')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='The image file you want to recognize must be 8x8 pixels')
    args = parser.parse_args()

    if len(_args) <= 0:
        parser.print_help()
    else:
        # 打开要识别的图片文件，,如果需要调试的时候可以显示在屏幕上
        image_file = args.file  # r'my_test_images/digit2.png'
        # image = Image.open(image_file)
        # image.show()

        # 发送图片数据给服务端，并获取返回的结果
        url = 'http://127.0.0.1:5000/prediction'
        files = {'file': open(image_file, 'rb')}
        response = requests.post(url, files=files)
        result = response.json()

        # 打印结果在控制台上，并保存为txt文件
        print("The handwritten number on the picture is:{}".format(result['prediction']))


# if __name__ == '__main__':
main(sys.argv[1:])
