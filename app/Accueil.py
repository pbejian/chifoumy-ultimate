#-------------------------------------------------------------------------------

import streamlit as st
import mediapipe as mp
#import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os
import pickle
import numpy as np
from chifoumy.ml_logic.registry import load_pipeline
from chifoumy.ml_logic.params import LOCAL_IMAGES_PATH
from chifoumy.interface.detection import picture_to_df
from chifoumy.interface.detection import espace

#-------------------------------------------------------------------------------

# print("⚠️")
# result = os.system("pwd")
# result = str(result)
# st.write(result)
# print("➡️ TOTO : ", result)

#-------------------------------------------------------------------------------

html_title = "<h2 style='color:#FF036A'>Chifoumy : pierre, feuille, ciseaux, python et Spock !</h2>"
st.markdown(html_title, unsafe_allow_html=True)

st.markdown("""
Vous connaissez tous le jeu « pierre, feuille ciseaux » (souvent appeler chifoumi).
Si ce n'est pas le cas, en voici une [description sur wikipedia](https://fr.wikipedia.org/wiki/Pierre-papier-ciseaux).
""", unsafe_allow_html=True)

st.markdown("""
Nous proposons ici une version étendue qui a été popularisée par la série « The Big Bang Theory ». Voici
[une vidéo du célèbre Sheldon Cooper](https://youtu.be/_PUEoDYpUyQ)
expliquant les règles du jeux à cinq positions. Nous avons remplacé le lézard par un python... car nous programmons en
[Python !](https://www.python.org).
""", unsafe_allow_html=True)

html_subtitle = "<h3 style='color:#44B7E3'>Testons la reconnaissance des cinq gestes.</h3>"
st.markdown(html_subtitle, unsafe_allow_html=True)

html_subtitle = "<p style='color:#000000'>NB - Pour une meilleure reconnaissance, approchez votre main de la camera .</p>"
st.markdown(html_subtitle, unsafe_allow_html=True)

picture = None
picture = st.camera_input(label=" ", disabled=False, key=666)
if picture:
    button1 = st.button("Tester la photo", key=1234)
    if button1:
        df = picture_to_df(picture)
        # st.write(type(df))
        if type(df) == type("toto"):
            st.write("Problème dans l'acquisition photo.")
        else:
            # Loading the pipeline
            my_pipeline = load_pipeline(spock=True)
            print("\n✅ spock model loaded from disk 🖖")

            # Apllying the pipeline to the new dataframe (scale and predict)
            target = my_pipeline.predict(df)
            target = target[0]

            html_pierre ="<h3 style='color:#44B7E3'>Votre geste : pierre</h3>"
            html_feuille ="<h3 style='color:#44B7E3'>Votre geste : feuille</h3>"
            html_ciseaux ="<h3 style='color:#44B7E3'>Votre geste : ciseaux</h3>"
            html_python ="<h3 style='color:#44B7E3'>Votre geste : python</h3>"
            html_spock ="<h3 style='color:#44B7E3'>Votre geste : Spock</h3>"
            chifoudict = {0: html_pierre, 1: html_feuille, 2: html_ciseaux,
                          3: html_python, 4: html_spock}
            html_gesture = chifoudict[target]
            st.markdown(html_gesture, unsafe_allow_html=True)

#-------------------------------------------------------------------------------
# Conclusion avec le lien vers les sources sur GitHub

espace(2)
st.markdown("""
    <hr>
""", unsafe_allow_html=True)
espace(2)
st.write("""
📝 Sources de l'application :
[https://github.com/pbejian/chifoumy-plus/](https://github.com/pbejian/chifoumy-plus/)
""")
#-------------------------------------------------------------------------------
