from ast import Str
import streamlit as st
import numpy as np
import os
from PIL import Image
from sklearn.cluster import KMeans
import requests
import base64
import random
import io
import cv2
from streamlit_chat import message
import openai
import threading
from streamlit_option_menu import option_menu
from streamlit_image_coordinates import streamlit_image_coordinates
import random

stability_engine_id = "stable-diffusion-xl-beta-v2-2-2"
stability_api_host = os.getenv('API_HOST', 'https://api.stability.ai')
stability_api_key = os.getenv("STABILITY_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model_id = os.getenv("OPENAI_MODEL_ID")


@st.cache_data(show_spinner='')
def auto_canny(image, sigma=0.33):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower, upper)

    edges = 255 - edges
    # to increase contrast
    edges = cv2.createCLAHE().apply(edges)
    return edges


@st.cache_data(show_spinner="Generating drawing...")
def getImageFromText(prompt: str):
    response = requests.post(
        f"{stability_api_host}/v1/generation/{stability_engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {stability_api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": f"a brightly colored drawing showing {prompt}",
                    "weight": 1
                },
                {
                    "text": f"person making a drawing",
                    "weight": -50
                }
            ],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30,
        },
    )
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    im = data["artifacts"][0]
    return base64.b64decode(im["base64"])

# image is an np array


@st.cache_data(show_spinner='')
def getKMeans(image, number_of_colors):
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)
    clf = KMeans(n_clusters=number_of_colors, n_init='auto')
    labels = clf.fit_predict(modified_image)
    # clusters is a mapping from cluster label to the list of pixel indices having that color
    clusters = dict()
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[int(label)] = []
        clusters[int(label)].append(i)
    return labels, clusters, np.array(clf.cluster_centers_, dtype='uint8')


@st.cache_data(show_spinner='Generating suggestions for your coloring session')
def get_completions(prompt: str):
    completionResponse = openai.Completion.create(
        model=openai_model_id, prompt=prompt.lower()+" ->", max_tokens=30, n=3)

    comps = []
    for c in completionResponse.choices:
        comps.append(c.text.split(".")[0] + ".") 
    return comps


def loadImage(prompt: str):
    image = Image.open(io.BytesIO(getImageFromText(prompt)))
    clustered = np.array(image)
    st.session_state.orig_shape = clustered.shape
    labels, clusters, cluster_centers = getKMeans(clustered, 10)
    st.session_state.color_labels = labels
    st.session_state.color_clusters = clusters
    st.session_state.cluster_centers = cluster_centers

    gs_image = image.convert('L')
    image_to_color = auto_canny(np.array(gs_image))
    image_to_color = cv2.cvtColor(image_to_color, cv2.COLOR_GRAY2RGB)
    st.session_state.image_to_color_flattened = image_to_color.reshape((-1, 3))
    st.session_state.colored_regions = set()
    st.session_state.coords = None
    st.session_state.num_images_generated += 1


def prompt_callback(key: str):
    loadImage(st.session_state[key])


def theme_callback(key: str):
    st.session_state.messages.append(
        {"role": "user", "content": st.session_state[key]})
    st.session_state.messages.append(
        {"role": "assistant", "content": "Here are a few suggestions for the content of your artwork."})
    st.session_state.messages.append(
        {"role": "prompt_options", "content": get_completions(st.session_state[key])})


def start_session():
    # remove the image from memory, because it will be displayed again, which we dont want
    st.session_state.pop('image_to_color_flattened', None)
    
    st.session_state.messages.append(
        {"role": "assistant", "content": "For your new session I have a few suggestions for a theme that you can explore"})
    st.session_state.messages.append(
        {"role": "theme_options", "content": themes})


st.set_page_config(
    page_title="Arther",
    page_icon="👋",
)

st.sidebar.markdown('<p class="font">Arther</p>', unsafe_allow_html=True)
with st.sidebar.expander("About Arther"):
    st.write("""
      Arther is a coloring app that generates a soothing artwork for you to color.          
     """)


st.sidebar.button('Start session', key='start_session',
                  type="primary", on_click=start_session)

themes = ['Nature and Landscapes', 'Mandalas and Geometric Patterns',
          'Fantasy and Mythology', 'Animals and Wildlife', 'Seasons and Holidays', 'Abstract and Psychedelic Art',
          'Artistic Styles and Art History', 'Cultural and Ethnic Art']

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
            "content": "Welcome to Arther! Your own personalized coloring book!"},
        {"role": "assistant", "content": "To start a coloring session, press the \'Start session\' button in the sidebar"}
    ]
    st.session_state.num_images_history = 0
    st.session_state.num_images_generated = 0

for i, m in enumerate(st.session_state['messages']):
    if m["role"] == "image":
        st.image(m["content"])
    elif m["role"] == "theme_options":
        option_menu(None, m["content"], on_change=theme_callback,
                    key=str(i) + '_theme_options')
    elif m["role"] == "prompt_options":
        option_menu(None, m["content"], on_change=prompt_callback, key=str(
            i) + '_prompt_options')
    else:
        message(m["content"], is_user=True if m["role"] ==
                "user" else False, key=str(i) + '_chat')

imageCursorJs = f"""
<script>
    function scroll(dummy_var_to_force_repeat_execution){{
        iFrames = parent.document.querySelectorAll('iframe')
        for (let i = 0; i < iFrames.length; i++) {{
            images = iFrames[i].contentDocument.body.querySelectorAll('img')
            for (let j = 0; j < images.length; j++) {{
              images[j].style.cursor = 'crosshair'
            }}
        }}
    }}
    scroll({random.random()})
</script>
"""
if 'image_to_color_flattened' in st.session_state:
    image_to_color_flattened = st.session_state.image_to_color_flattened
    image = Image.fromarray(
        image_to_color_flattened.reshape(st.session_state.orig_shape))
    streamlit_image_coordinates(image, key="coords")
    st.components.v1.html(imageCursorJs)
    col_progress = int(
        100*(len(st.session_state.colored_regions)/len(st.session_state.color_clusters)))
    st.progress(col_progress, text=f'{col_progress}% completed')
    if col_progress >= 100:
        st.balloons()
        if st.session_state.num_images_history < st.session_state.num_images_generated:
            st.session_state.messages.append(
                {"role": "image", "content": image})
            st.session_state.num_images_history += 1
            congrats_msg = "Congratulations!! You have completed a coloring session. You can start another one clicking 'Start session' from the sidebar"
            message(congrats_msg, is_user=False)
            st.session_state.messages.append(
                {"role": "assistent", "content": congrats_msg})
    elif st.session_state.coords:
        idx = st.session_state.coords["y"] * \
            st.session_state.orig_shape[0] + st.session_state.coords["x"]
        if st.session_state.color_labels[idx] not in st.session_state.colored_regions:
            pixels = st.session_state.color_clusters[st.session_state.color_labels[idx]]
            for pIdx in pixels:
                image_to_color_flattened[pIdx] = st.session_state.cluster_centers[st.session_state.color_labels[idx]]
            st.session_state.colored_regions.add(
                st.session_state.color_labels[idx])
            st.experimental_rerun()
