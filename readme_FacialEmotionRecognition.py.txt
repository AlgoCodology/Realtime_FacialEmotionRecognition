In order to run the code from console follow these steps:

1. Go to the directory which contains the FacialEmotionRecognition.py file
2. Make sure the fer2013.csv file containing the data is also located in that
   directory.
3. Run the code by using following command:
	python FacialEmotionRecognition.py fer2013.csv
4. The code will run training the model specified in the FacialEmotionRecognition.py
5. After training is done the model and its history will be stored in the
   current directory.
6. All the produced figures will be stored in the current directory.
7. If an already trained model should be loaded, lines 228 until 247
   should be commented out. Then lines 250 until 251 and line 256 
   should be uncommented.
   Specify the name of the model that should be loaded in line 250.
   Make sure the directory containing the model is in the current working directory and 
   has the same name as the name specified in line 250.
   Run the code as mentioned in point 3.
