# An Online Captcha Recognition Server

This HKU MSc CS final project, a cloud service of captcha recognition system based on deep learning. We provide both Web API and UI, as well as Chrome Extension support for convience. You may refer to our project [homepage](https://i.cs.hku.hk/~msp18012/) and more details from [script](https://github.com/CaptainDuke/image-classify-server/blob/master/res/msp18012.pdf)

We have uploaded our docker image into Dockerhub, which has been downloaded for 10k+ times. Have a try: 
```
docker pull taylorliang/mycaptcha
```
Sample captchas: <br>
![captcha](https://github.com/CaptainDuke/image-classify-server/blob/master/res/Captchas.png)

It's is written in Django along with Tensorflow.<br>
We choose LSTM + CTC model to recognize captchas with variable length.<br>
The web interface is made using [materializecss](http://materializecss.com/) and [jQuery](https://jquery.com/)<br>
The Chrome Exetension interface is made using [screenshot-capture](https://github.com/simov/screenshot-capture), sending post via Ajax.

Drag and Drop !
- Chrome Extension interface, the result will be copied to the user clipboard automatically<br>![chrome](https://github.com/CaptainDuke/image-classify-server/blob/master/res/ChromeUser.png)
- Web UI interface<br>![webui](https://github.com/CaptainDuke/image-classify-server/blob/master/res/WebUI.png)![webui](https://github.com/CaptainDuke/image-classify-server/blob/master/res/WebUIOutput.png)


## Usage

To run the server on localhost:

```
$ pip3 install -r requirements.txt
$ python3 manage.py collectstatic
$ python3 manage.py runserver
```

## Using Retrained Model
* Retrain the model using your images. Refer [here](https://www.tensorflow.org/tutorials/image_retraining).
* Replace the generated graph and label files in `/classify_image/inception_model/`
* Deploy the Django project

## Cloud Deployment
 Aiming at effectively development and clearness of work assignment, we separate our system into two main parts. The first part is frontend, including user interface and Ajax engine, while another part is backend, which consists of Django and recognition model.<br>
![arc](https://github.com/CaptainDuke/image-classify-server/blob/master/res/arc.png)

 In order to improve the availability, throughput and scalability of our captcha recognition service, we deploy our captcha recognition server on a distributed computation cluster and choose Kubernetes to manage our captcha recognition application.<br>
![arc](https://github.com/CaptainDuke/image-classify-server/blob/master/res/k8s.png)



## Reference:
- https://pypi.org/project/captcha/
- https://github.com/DeepBlueCitzenService/Tensorflow-Server
- https://github.com/simov/screenshot-capture
- https://github.com/PatrickLib/captcha_recognize
- https://github.com/jimmyleaf/ocr_tensorflow_cnn
