
# VGG16 based Image classifier

We have trained a VGG 16 pre-trained model on Pascal VOC 2012 dataset. The initial VGG16 model is set to classify 1000 labels, however the Pascal VOC dataset has only 20 classes namely
`person, bottle, aeroplane, bird, boat, dog, pottedplant, cat, sofa, bicycle, chair, car, train, cow, horse, tvmonitor, diningtable, bus, sheep, motorbike`

We have inported the VGG 16 model, removed the last prediction layer and replaced it by another prediction layer for predicting 20 classes. Additionally we have trained the model for the new last layer only, and it resulted in about 71% accuracy on validation dataset.

Once the model has trained, we downloaded the weights file and developed a python web app using `streamlit`. This app is supposed to take a jpeg image as input, preprocess it as per VGG16 architecture requirements and then classify the said image.

Then we created 2 additional files in our working directory namely, `Dockerfile` and `requirements.txt` to further contanerize our application.

We created the docker image by running the following command - `docker build -t vgg16classifier_image .`

We created the docker container by running the following command - `docker run --name vgg16_classifier -p 8501:8501 vgg16_classifier_image`

Weights File is present at -- [https://drive.google.com/file/d/1w7VdUlaefuIXs4cI5NRImNKEWFJsSusB/view?usp=sharing]
