import tensorflow as tf
from typing import List
import cv2
import os 

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]: 
    #print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
    
def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
import os

def load_data(path: str): 
    # Decode the path from a tensor to a string
    path = bytes.decode(path.numpy())
    
    # Extract the file name without extension using os.path
    file_name = os.path.splitext(os.path.basename(path))[0]
    
    # Construct paths for video and alignment files
    video_path = os.path.join('C://Users//KIRAN//Documents//ML//RESEARCH PAPER- BASE//LipNet//app//data//s1', f'{file_name}.mpg')
    alignment_path = os.path.join('C://Users//KIRAN//Documents//ML//RESEARCH PAPER- BASE//LipNet//app//data//alignments//s1', f'{file_name}.align')
    
    # Load video frames and alignments with error handling
    try:
        frames = load_video(video_path)
    except FileNotFoundError:
        print(f"Error: The video file '{video_path}' does not exist.")
        frames = None  # Or handle accordingly

    try:
        alignments = load_alignments(alignment_path)
    except FileNotFoundError:
        print(f"Error: The alignment file '{alignment_path}' does not exist.")
        alignments = None  # Or handle accordingly
    
    return frames, alignments
