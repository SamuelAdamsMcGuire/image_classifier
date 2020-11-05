# ImageClassifier

Use AI to automatically distinguish between *apples*, *oranges* and *bananas* in real time using your webcam


## usage
-install environment requirements found in the environment.yml file with the following command in the bash shell:
```
conda env create -f environment.yml
```

-execute file from root folder
```python
python real_time_classifier.py pics
```
show webcam one of the 3 types of fruit or take pictures with webcam (space bar) and save in folder (in this case 'pics')
exit program with `q`


## Project 
- Built a deep learning pipeline with Keras (and other tools) that classifies images of objects. 
- Made own data set taking pictures of the various fruits and preprocessed the pictures
	- decrease file size 
	- scaling
	- cropping
- Data Augmentation used to increase the training data set
- Used DenseNet121 pre-trained network and added final layers that were trained for image classification.
- Used OpenCV to read images directly from webcam
- Classification result and probability given in real time on screen

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
