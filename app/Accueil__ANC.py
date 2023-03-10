 #===============================================================================

import matplotlib.pyplot as plt
import streamlit as st
import mediapipe as mp
import pandas as pd
from PIL import Image
import numpy as np
#import random
import os
#import cv2
from PIL import Image
#import include
#from include import create_key
from chifoumy.ml_logic.params import LOCAL_IMAGES_PATH

import subprocess

print("----"*5)

result = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))

print("----"*5)

result = subprocess.run(['pwd'], stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))

print("----"*5)

#===============================================================================

#IMAGE_PATH = "/images/"
# get latest model_pipeline version
images_directory = os.path.join(LOCAL_IMAGES_PATH)

#===============================================================================

html_title = "<h1 style='color:#FF036A;text-align:left;font-size:90px'>Projet Chifoumy</h1>"
st.markdown(html_title, unsafe_allow_html=True)
image_path = images_directory + "/chifoumi__ter.jpg"
chifoumi_image = Image.open(image_path)
picture = st.image(chifoumi_image, width=600)
#button_play = st.button("Jouer")

#===============================================================================

#st.write(create_key())
#st.write(create_key())
#st.write(create_key())
#st.write(create_key())
