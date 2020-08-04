# GesturesTrainApi

Flask Api to be integrated with android and ios 2D games like pacman, snake2d etc. API facilitates real time training of lite models initiated via POST request so that users can choose their own gesture for controls and click images and initiate training of the model which is deployed via API. 



Endpoint for post requests to train a model for custom gesture classification(four classes) '/train'. 
To start the flask server run file app.py or use deployed api : ~~ec2-184-72-134-37.compute-1.amazonaws.com:8080/train~~ (instance is now terminated). Flask server will be up and running.

Body for post request should contain a json of following format : 

{

  "images" : list of urls of images,
  
  "labels" : list of labels of urls provided (0/1/2/3),
  
  "epochs" : No. of epochs to be trained for
  
}


Trained model will be uploaded to firebase storage and its url will be sent as response which can be used to download the model.

Note: Populate the config dictionary in app.py if you wanna use firebase storage.
