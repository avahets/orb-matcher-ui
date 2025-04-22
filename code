import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="ORB Image Matcher", layout="centered")
st.title("🔍 ORB Keypoint Matcher")
st.write("Upload two images (left and right) to visualize keypoint matching using ORB.")

left_file = st.file_uploader("Upload Left Image", type=["jpg", "jpeg", "png"], key="left")
right_file = st.file_uploader("Upload Right Image", type=["jpg", "jpeg", "png"], key="right")

def read_image(file):
    img = Image.open(file).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

if left_file and right_file:
    left = read_image(left_file)
    right = read_image(right_file)

    orb = cv2.ORB_create()
    kp_left, des_left = orb.detectAndCompute(left, None)
    kp_right, des_right = orb.detectAndCompute(right, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_left, des_right)
    best = sorted(matches, key=lambda x: x.distance)[:10]

    matched_img = cv2.drawMatches(left, kp_left, right, kp_right, best, None, matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    st.image(matched_img_rgb, caption="Top 10 ORB Keypoint Matches", use_column_width=True)
else:
    st.info("Please upload both left and right images to see the result.")
