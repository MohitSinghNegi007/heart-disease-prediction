import streamlit as st

from ui import CSS

st.set_page_config(page_title="CardioCheck · Heart Disease Risk", page_icon="🫀", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

page = st.navigation([
    st.Page("views/lifestyle.py", title="Lifestyle Check", icon="🏃", default=True),
    st.Page("views/clinical.py", title="Clinical Check", icon="🩺"),
], position="top")
page.run()

st.markdown('<div class="footer">Developed by Mohit Singh Negi © 2026 · ML Project</div>',
            unsafe_allow_html=True)
