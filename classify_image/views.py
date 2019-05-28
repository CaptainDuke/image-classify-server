import io
import os

from base64 import b64decode
import tensorflow as tf
from PIL import Image
from django.core.files.temp import NamedTemporaryFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

MAX_K = 10

TF_GRAPH = "{base_path}/inception_model/graph.pb".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
TF_LABELS = "{base_path}/inception_model/labels.txt".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
check_point_dir = "{base_path}/captcha_letter/captcha_train".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
# print(check_point_dir)
from datetime import datetime
import classify_image.captcha_letter.config as config
import classify_image.captcha_letter.captcha_model as captcha
from tensorflow.python.platform import gfile
import numpy as np
import os.path
import sys
import argparse



IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT

CHAR_SETS = config.CHAR_SETS
CLASSES_NUM = config.CLASSES_NUM
CHARS_NUM = config.CHARS_NUM

FLAGS = None


def one_hot_to_texts(recog_result):
    texts = []
    for i in range(recog_result.shape[0]):
        index = recog_result[i]
        texts.append(''.join([CHAR_SETS[i] for i in index]))
    return texts


def input_data(image):

    #batch_size = len(file_list)
    #images = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH], dtype='float32')
    files = []
    i = 0

    image = Image.open(image)

    image_gray = image.convert('L')
    image_resize = image_gray.resize(size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    image.close()
    input_img = np.array(image_resize, dtype='float32')
    input_img = np.multiply(input_img.flatten(), 1./255) - 0.5

    return input_img


def myload_graph():

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_image = tf.placeholder(tf.float32, shape=[IMAGE_HEIGHT*IMAGE_WIDTH])
        input_filename = ''
        # images = tf.constant(input_image)
        logits = captcha.inference(input_image, keep_prob=1)
        result = captcha.output(logits)
        saver = tf.train.Saver()
        sess = tf.Session()
        print(check_point_dir)
        saver.restore(sess, tf.train.latest_checkpoint(check_point_dir))
        print(tf.train.latest_checkpoint(check_point_dir))
    return sess, result, input_image

SESS2, GRAPH_TENSOR2 , INPUT_IMAGE= myload_graph()


def load_graph():
    sess = tf.Session()
    with tf.gfile.FastGFile(TF_GRAPH, 'rb') as tf_graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(tf_graph.read())
        tf.import_graph_def(graph_def, name='')
    label_lines = [line.rstrip() for line in tf.gfile.GFile(TF_LABELS)]
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    return sess, softmax_tensor, label_lines



# SESS, GRAPH_TENSOR, LABELS = load_graph()

# my_classify_api used for captcha
@csrf_exempt
def classify_api(request):
    data = {"success": False}
    print("my_classify api called")
    if request.method == "POST":
        tmp_f = NamedTemporaryFile()

        if request.FILES.get("image", None) is not None:
            image_request = request.FILES["image"]
            image_bytes = image_request.read()
            image = Image.open(io.BytesIO(image_bytes))
            image.save(tmp_f, image.format)
        elif request.POST.get("image64", None) is not None:
            base64_data = request.POST.get("image64", None).split(',', 1)[1]
            plain_data = b64decode(base64_data)
            tmp_f.write(plain_data)

        print("calling mytf_classify")
        classify_result = mytf_classify(tmp_f, int(request.POST.get('k', MAX_K)))
        tmp_f.close()
        print(classify_result)
        if classify_result:
            data["success"] = True
            data["confidence"] = {}

            #data["confidence"][res[0]] = float(res[1])
            data["confidence"][classify_result[0]] = float(1)


    return JsonResponse(data)


def mytf_classify(image_file, k=MAX_K):
    result = list()
    print("mytf_classify called")
    #image_data = tf.gfile.FastGFile(image_file.name, 'rb').read()
    image_data = input_data(image_file)

    predictions = SESS2.run(GRAPH_TENSOR2, feed_dict={INPUT_IMAGE: image_data})
    text = one_hot_to_texts(predictions)
    top_k = predictions.argsort()[-k:][::-1]

    return text



@csrf_exempt
# def classify_api(request):
#     data = {"success": False}
#
#     if request.method == "POST":
#         tmp_f = NamedTemporaryFile()
#
#         if request.FILES.get("image", None) is not None:
#             image_request = request.FILES["image"]
#             image_bytes = image_request.read()
#             image = Image.open(io.BytesIO(image_bytes))
#             image.save(tmp_f, image.format)
#         elif request.POST.get("image64", None) is not None:
#             base64_data = request.POST.get("image64", None).split(',', 1)[1]
#             plain_data = b64decode(base64_data)
#             tmp_f.write(plain_data)
#
#         classify_result = tf_classify(tmp_f, int(request.POST.get('k', MAX_K)))
#         tmp_f.close()
#
#         if classify_result:
#             data["success"] = True
#             data["confidence"] = {}
#             for res in classify_result:
#                 data["confidence"][res[0]] = float(res[1])
#                 # data["confidence"]['TEST'] = float(res[1])
#                 # break
#
#     return JsonResponse(data)


def classify(request):
    return render(request, 'classify.html', {})

def myclassify(request):
    return render(request, 'myclassify.html', {})


# noinspection PyUnresolvedReferences
def tf_classify(image_file, k=MAX_K):
    result = list()

    image_data = tf.gfile.FastGFile(image_file.name, 'rb').read()

    predictions = SESS.run(GRAPH_TENSOR, {'DecodeJpeg/contents:0': image_data})
    predictions = predictions[0][:len(LABELS)]
    top_k = predictions.argsort()[-k:][::-1]
    for node_id in top_k:
        label_string = LABELS[node_id]
        score = predictions[node_id]
        result.append([label_string, score])

    return result
