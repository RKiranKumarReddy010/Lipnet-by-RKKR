# Import all of the dependencies
import streamlit as st
import os
import imageio
import base64
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the Streamlit app as wide 
st.set_page_config(layout='wide')

def load_gif(file_path):
    with open(file_path, "rb") as file:
        contents = file.read()
        data_url = base64.b64encode(contents).decode("utf-8")
    return f'<img src="data:image/gif;base64,{data_url}" width="300" height="300" alt="GIF">'

# Setup the sidebar
with st.sidebar: 
    # Path to your local GIF file
    gif_path = r"C:\Users\KIRAN\Documents\ML\RESEARCH PAPER- BASE\LipNet\app\5eeea355389655.59822ff824b72.gif"  # Update with your actual path

    # Display the GIF in the Streamlit app
    st.markdown(load_gif(gif_path), unsafe_allow_html=True)
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet BY MotionBlurr') 

# File uploader for user to input their own video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary location
    temp_video_path = "uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Generate two columns 
    col1, col2 = st.columns(2)

    # Rendering the video 
    with col1: 
        st.info('The video below displays the uploaded video in mp4 format')
        st.video(temp_video_path)

        # Convert the uploaded video using ffmpeg (if needed)
        os.system(f'ffmpeg -i "{temp_video_path}" -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        with open('test_video.mp4', 'rb') as video:
            video_bytes = video.read()
            st.video(video_bytes)

    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        
        # Load data for prediction
        video_data, annotations = load_data(tf.convert_to_tensor(temp_video_path))

        # Ensure video_data is a 4D NumPy array (num_frames, height, width, channels)
        if isinstance(video_data, np.ndarray) and video_data.ndim == 4:
            # Convert to list of images for GIF creation
            images = [video_data[i] for i in range(video_data.shape[0])]
            
            # Create an animation from the video frames
            imageio.mimsave('animation.gif', images, fps=10)
            st.image('animation.gif', width=400) 
        else:
            st.error("Invalid video data format. Please check your input.")

        st.info('This is the output of the machine learning model as tokens')
        
        model = load_model()
        yhat = model.predict(tf.expand_dims(video_data, axis=0))
        
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
else:
    st.info("Please upload a video to get started.")
