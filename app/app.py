# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import base64
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

def load_gif(file_path):
    with open(file_path, "rb") as file:
        contents = file.read()
        data_url = base64.b64encode(contents).decode("utf-8")
    return f'<img src="data:image/gif;base64,{data_url}" width="300" height="300" alt="GIF">'


# Setup the sidebar
with st.sidebar: 
    # Path to your local GIF file
    gif_path = "C://Users//KIRAN//Documents//ML//RESEARCH PAPER- BASE//LipNet//app//5eeea355389655.59822ff824b72.gif"  # Update with your actual path

    # Display the GIF in the Streamlit app
    st.markdown(load_gif(gif_path), unsafe_allow_html=True)
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet BY MotionBlurr') 
# Generating a list of options or videos 
options = os.listdir('C://Users//KIRAN//Documents//ML//RESEARCH PAPER- BASE//LipNet//app//data//s1')
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('C://Users//KIRAN//Documents//ML//RESEARCH PAPER- BASE//LipNet//app//data//s1//', selected_video)
        path = os.path.join(r'C:\Users\KIRAN\Documents\ML\RESEARCH PAPER- BASE\LipNet', r'app\data\s1', selected_video)
        os.system(f'ffmpeg -i "{path}" -vcodec libx264 test_video.mp4 -y')
        # Rendering inside of the app
        with open('test_video.mp4', 'rb') as video:
            video_bytes = video.read()
            st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
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
        