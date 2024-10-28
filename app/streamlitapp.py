# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
import numpy as np

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://i.pinimg.com/550x/51/45/66/5145668e8e638ae7341fa408a76a0fbf.jpg')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning paper.')

st.title('LipNet Testing') 
# Generating a list of options or videos 
options = os.listdir('C://Users//KIRAN//Documents//ML//RESEARCH PAPER- BASE//LipNet//app//data//s1')
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 
    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('C://Users//KIRAN//Documents//ML//RESEARCH PAPER- BASE//LipNet//app//data//s1', selected_video)
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
        else:
            os.system(f'ffmpeg -i "{file_path}" -c:v libx264 -preset medium -crf 23 -c:a aac test_video.mp4 -y')
        #os.system(f'ffmpeg -i {file_path} -c:v libx264 -preset medium -crf 23 -c:a aac output.mp4')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the model sees when making a prediction')
        video = [np.random.rand(46, 140, 1) for _ in range(75)]
        imageio.mimsave('animation.gif', video, fps=30)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)