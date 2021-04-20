# CSI 5325 Final Project

This project is an attempted solution to the Kaggle problem 
["Indoor Logation & Navigation"](https://www.kaggle.com/c/indoor-location-navigation).
It uses a Long Short-Term Memory (LSTM) model to analyze a 
[pre-processed version](https://kaggle.com/devinanzelmo/indoor-navigation-and-location-wifi-features) 
of the data which only contains Wi-Fi features of a certain strength and returns the results.

This project was completed for course credit for CSI 5325 at Baylor University, taught by
[Dr. Pablo Rivas](https://rivas.ai/).

I constructed this model using the Keras API for Python. This repository contains the code
which produces the model and my report, written in LaTeX.

In order to run the project, you will need to download the data files from the link given above and
place them in a data/ directory with two subdirectories, test/ and train/. Next, you will need to initialize
your local environment by installing the Python dependencies using pip:

```commandline
pip install -r requirements.txt
```

I recommend first installing and using a virtual environment as follows:

```commandline
python -m venv venv
source venv/bin/activate
```

Finally, you can run the project as follows:

```commandline
python main.py
```