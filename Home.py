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
from streamlit_image_coordinates import streamlit_image_coordinates

stability_engine_id = "stable-diffusion-xl-beta-v2-2-2"
stability_api_host = os.getenv('API_HOST', 'https://api.stability.ai')
stability_api_key = os.getenv("STABILITY_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model_id = os.getenv("OPENAI_MODEL_ID") 


class openAiChatThread(threading.Thread):
    def __init__(self, prompt):
        threading.Thread.__init__(self)
        self.prompt = prompt
        self.resp = ""

    def run(self):
        completionResponse = openai.Completion.create(
            model=openai_model_id,
            prompt=self.prompt,
            max_tokens=50)
        self.resp = completionResponse.choices[0].text


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


@st.cache_data(show_spinner='Generating suggestions for your artwork')
def getCompletions(prompts):
    reqThreads = []
    for p in prompts:
        t = openAiChatThread(p)
        t.start()
        reqThreads.append(t)

    comps = []
    for t in reqThreads:
        t.join()
        comps.append(t.resp.split(".")[0] + ".")
    return comps


def user_msg_callback(key: str):
    if st.session_state[key]: 
      st.session_state.messages.append({"role": "user", "content": st.session_state[key]})
      # remove last user message placeholder
      st.session_state.user_msg_placeholders[-1].empty()

      st.session_state.show_user_text_input = False
      
      # show image to color
      image = Image.open(io.BytesIO(getImageFromText(st.session_state[key])))
      clustered = np.array(image)
      st.session_state.orig_shape = clustered.shape
      labels, clusters, cluster_centers = getKMeans(clustered, 10)
      st.session_state.color_labels = labels
      st.session_state.color_clusters = clusters
      st.session_state.cluster_centers = cluster_centers
      gs_image = image.convert('L')
      image_to_color = auto_canny(np.array(gs_image))
      image_to_color = cv2.cvtColor(image_to_color, cv2.COLOR_GRAY2RGB)
      image_to_color_flattened = image_to_color.reshape((-1, 3))
      st.session_state.image_to_color_flattened = image_to_color_flattened
      st.session_state.colored_regions = set()
      st.session_state.coords = None
      st.session_state.num_images_generated += 1


def themeCallback():

    # remove the image from memory, because it will be displayed gaian, which we dont want
    st.session_state.pop('image_to_color_flattened', None)

    if st.session_state.theme:
        st.session_state.messages.append({"role": "user", "content": st.session_state.theme})
        comps = getCompletions(
            generatePromptsFromTheme(st.session_state.theme))
        for c in comps:
            if "your" in c:
                comps.remove(c)
        if len(comps) > 5:
            comps = random.sample(comps, 5)

        st.session_state.messages.append(
            {"role": "assistant", "content": "Here are some suggestions for the content of your artwork. You can either copy and paste one of these or input an entirely new artwork description."})
        for i, c in enumerate(comps):
            st.session_state.messages.append(
                {"role": "assistant", "content": str(i+1) + ". " + c})
        st.session_state["show_user_text_input"] = True


def generatePromptsFromTheme(theme: Str):
    # each context is a prefix and an optional example intended to be used after the theme

    contexts = [("Create an abstract artwork that represents",), ("Paint a self-portrait that reflects",), ("Draw a nature scene that represents", "such as a serene forest or a calming beach"),
                ("Create a visual representation of", "incorporating images and words that inspire you"), (
                    "Draw a mandala and fill it with patterns and colors that represents", ),
                ("Create a mixed media artwork using magazine cutouts, photographs, and paint to express",), (
                    "Draw a still life arrangement of objects representing", "capturing their details and textures"),
                ("Create artwork that illustrates", "combining images, colors, and words to create a visual representation.")]

    prompts = []
    for c in contexts:
        if len(c) == 1:
            p = c[0] + " " + theme + "."
        else:
            p = c[0] + " " + theme + ", " + c[1] + "."
        prompts.append(p)

    return prompts

st.set_page_config(
    page_title="Arther",
    page_icon="👋",
)

st.sidebar.markdown('<p class="font">Arther</p>', unsafe_allow_html=True)
with st.sidebar.expander("About Arther"):
    st.write("""
      Arther is a coloring app that generates a soothing artwork for you to color          
     """)



themes = ['', 'your current emotions', 'your inner state', 'a sense of peace and tranquility', 'your goals and aspirations', 'a recent challenge you have overcome',
          'a fear or anxiety you are currently experiencing', 'a significant life event or transition you have been through', 'positive affirmation or mantra that inspires and uplifts you']
sIdx = 0
if "theme" in st.session_state:
    sIdx = themes.index(st.session_state.theme)
st.sidebar.selectbox('themes', index=sIdx, options=themes,
                     key="theme", on_change=themeCallback)

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Arther!"},
        {"role": "assistant", "content": "To get started please select a theme that you'd like to explore in today's session!"}
    ]
    st.session_state.num_images_history = 0
    st.session_state.num_images_generated = 0

if 'user_msg_placeholders' not in st.session_state:
    st.session_state.user_msg_placeholders = []

for i, m in enumerate(st.session_state['messages']):
    if m["role"] == "image":
      st.image(m["content"])
    else:
      message(m["content"], is_user=True if m["role"] == "user" else False, key=str(i) + '_chat')

# render user input text box
if "show_user_text_input" in st.session_state and st.session_state.show_user_text_input:
    user_msg_placeholder = st.empty()
    with user_msg_placeholder.container():
        key = "user_msg_" + str(len(st.session_state.messages))
        st.text_input(key, key=key, on_change=user_msg_callback,
                      args=(key,), label_visibility="hidden")
    st.session_state.user_msg_placeholders.append(user_msg_placeholder)


if 'image_to_color_flattened' in st.session_state:
    image_to_color_flattened = st.session_state.image_to_color_flattened
    image = Image.fromarray(
        image_to_color_flattened.reshape(st.session_state.orig_shape))
    streamlit_image_coordinates(image, key="coords")
    col_progress = int(
        100*(len(st.session_state.colored_regions)/len(st.session_state.color_clusters)))
    st.progress(col_progress, text=f'{col_progress}% completed')
    if col_progress >= 100:
        st.balloons()
        if st.session_state.num_images_history < st.session_state.num_images_generated:
          st.session_state.messages.append({"role": "image", "content": image})
          st.session_state.num_images_history += 1
          congrats_msg = "Congratulations!! You have completed a coloring challenge. You can start another one by picking a different theme from the sidebar"
          message(congrats_msg, is_user=False)
          st.session_state.messages.append({"role": "assistent", "content": congrats_msg})
    elif st.session_state.coords:
        idx = st.session_state.coords["y"]*st.session_state.orig_shape[0] + st.session_state.coords["x"]
        if st.session_state.color_labels[idx] not in st.session_state.colored_regions:
            pixels = st.session_state.color_clusters[st.session_state.color_labels[idx]]
            for pIdx in pixels:
                image_to_color_flattened[pIdx] = st.session_state.cluster_centers[st.session_state.color_labels[idx]]
            st.session_state.colored_regions.add(
                st.session_state.color_labels[idx])
            st.experimental_rerun()
