import streamlit as st
from mps_backend import main

st.title("Handwriting Synthesis")
text=st.text_area(label="Enter Text for handwriting generation")
clicked=st.button("Submit")
images=[]
if(clicked):
    images=main("snap\IAM_weights\IAM_weights\IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG\checkpoint-iteration175000.pth"
                 , "examples", style_loc="snap\IAM_weights\IAM_weights\IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG\\test_styles_175000.pkl",
                 text=text,num_inst=5)
for i in images:
    st.image(i)
