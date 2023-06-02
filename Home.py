import streamlit as st
import numpy as np
import os
from PIL import Image
from PIL import ImageColor
from sklearn.cluster import KMeans
import requests
import base64
import random
import io
import cv2
from streamlit_chat import message
import openai
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

    # gaussian blur to remove aliasing
    return cv2.GaussianBlur(edges, (3, 3), 0)


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
    # clf = KMeans(n_clusters=number_of_colors, n_init='auto')
    clf = KMeans(n_init='auto')
    labels = clf.fit_predict(modified_image)
    cluster_centers = np.array(clf.cluster_centers_, dtype='uint8')
    # clusters is a mapping from cluster label to the list of pixel indices having that color
    clusters = dict()
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[int(label)] = []
        clusters[int(label)].append(i)
        modified_image[i] = cluster_centers[label]
    return labels, clusters, cluster_centers, modified_image


def get_completions(prompt: str):
    with st.spinner(''):
        completionResponse = openai.Completion.create(
            model=openai_model_id, prompt=prompt.lower()+" ->", max_tokens=30, n=3)

    comps = []
    for c in completionResponse.choices:
        comps.append(c.text.split(".")[0] + ".")
    return comps


def load_image(prompt: str, session_tab_idx: int):
    image = Image.open(io.BytesIO(getImageFromText(prompt)))
    to_be_clustered = np.array(image)
    labels, clusters, cluster_centers, clustered_flat = getKMeans(
        to_be_clustered, 10)
    gs_image = image.convert('L')
    image_to_color = auto_canny(np.array(gs_image))
    image_to_color = cv2.cvtColor(image_to_color, cv2.COLOR_GRAY2RGB)

    st.session_state.session_tabs_state[session_tab_idx]['image_to_color_flattened'] = image_to_color.reshape(
        (-1, 3))
    st.session_state.session_tabs_state[session_tab_idx]['image_clustered_flattened'] = clustered_flat
    st.session_state.session_tabs_state[session_tab_idx]['orig_shape'] = to_be_clustered.shape
    st.session_state.session_tabs_state[session_tab_idx]['color_labels'] = labels
    st.session_state.session_tabs_state[session_tab_idx]['color_clusters'] = clusters
    st.session_state.session_tabs_state[session_tab_idx]['cluster_centers'] = cluster_centers
    st.session_state.session_tabs_state[session_tab_idx]['colored_regions'] = set(
    )

    st.session_state[f'{session_tab_idx}_coords'] = None


def prompt_callback(key: str):
    tab_idx = int(key.split("_")[0])
    if not st.session_state.session_tabs_state[tab_idx]['in_progress']:
        st.error('This session is already complete!')
        return

    prompt_set_idx = int(key.split("_")[1])

    prompt_options = st.session_state.session_tabs_state[tab_idx]['prompt_options'][prompt_set_idx][1]
    last_selected_prompt_idx = 0
    for i, p in enumerate(prompt_options):
        if p == st.session_state[key]:
            last_selected_prompt_idx = i
            break
    st.session_state.session_tabs_state[tab_idx]['prompt_options'][prompt_set_idx] = (
        last_selected_prompt_idx, prompt_options)
    st.session_state.session_tabs_state[tab_idx]['messages'].append(
        {'role': 'user', 'content': st.session_state[key]})
    st.session_state.session_tabs_state[tab_idx]['messages'].append(
        {'role': 'assitant', 'content': "Arther's sketch is below. To paint, \
            select a color from the color-picker just below the image, and click on an uncolored \
            region in the sketch. This will fill all the regions that share the color with the clicked point."})

    load_image(st.session_state[key], tab_idx)


def theme_callback(key: str):
    tab_idx = int(key.split("_")[0])
    if not st.session_state.session_tabs_state[tab_idx]['in_progress']:
        # the session is already complete
        st.info('This session is already complete!')
        return

    last_selected_theme_idx = 0
    for i, t in enumerate(themes):
        if t == st.session_state[key]:
            last_selected_theme_idx = i
            break
    st.session_state.session_tabs_state[tab_idx]['last_selected_theme_idx'] = last_selected_theme_idx

    st.session_state.session_tabs_state[tab_idx]['prompt_options'].append(
        (0, get_completions(st.session_state[key])))

    st.session_state.session_tabs_state[tab_idx]['messages'].append(
        {"role": "user", "content": st.session_state[key]})
    st.session_state.session_tabs_state[tab_idx]['messages'].append(
        {"role": "assistant", "content": "Here are a few suggestions for the contents of your sketch."})

    # for prompt options, include the index of the most recently added
    st.session_state.session_tabs_state[tab_idx]['messages'].append(
        {"role": "prompt_options", "content": len(st.session_state.session_tabs_state[tab_idx]['prompt_options'])-1})


def start_session():
    # add a new session tab
    st.session_state.session_tabs_state.append({'messages':
                                                [{'role': 'assistant', 'content': 'For your new session I have a few suggestions for a theme that you can explore.'},
                                                 {'role': 'theme_options'}],
                                                'in_progress': True,
                                                'last_selected_theme_idx': 0,
                                                'prompt_options': []
                                                })


def color_region(coords, tab_idx, hex_color):
    color_clusters = st.session_state.session_tabs_state[tab_idx]['color_clusters']
    color_labels = st.session_state.session_tabs_state[tab_idx]['color_labels']
    image_to_color_flattened = st.session_state.session_tabs_state[
        tab_idx]['image_to_color_flattened']
    idx = coords['y'] * \
        st.session_state.session_tabs_state[tab_idx]['orig_shape'][0] + coords['x']

    if color_labels[idx] in st.session_state.session_tabs_state[tab_idx]['colored_regions']:
        return
    pixels = color_clusters[color_labels[idx]]
    for pIdx in pixels:
        image_to_color_flattened[pIdx] = list(
            ImageColor.getcolor(hex_color, 'RGB'))
    st.session_state.session_tabs_state[tab_idx]['colored_regions'].add(
        color_labels[idx])
    st.experimental_rerun()


def render_image(tab, tab_idx: int):
    if 'image_to_color_flattened' not in st.session_state.session_tabs_state[tab_idx]:
        return
    with tab:
        image_to_color_flattened = st.session_state.session_tabs_state[
            tab_idx]['image_to_color_flattened']
        orig_shape = st.session_state.session_tabs_state[tab_idx]['orig_shape']
        num_colored_regions = len(
            st.session_state.session_tabs_state[tab_idx]['colored_regions'])
        num_color_clusters = len(
            st.session_state.session_tabs_state[tab_idx]['color_clusters'])

        image = Image.fromarray(image_to_color_flattened.reshape(orig_shape))
        streamlit_image_coordinates(image, key=f'{tab_idx}_coords')
        st.components.v1.html(imageCursorJs)
        color = st.color_picker('Pick A Color')
        col_progress = int(100*num_colored_regions/num_color_clusters)
        st.progress(col_progress, text=f'{col_progress}% completed')
        if col_progress >= 100:
            if st.session_state.session_tabs_state[tab_idx]['in_progress']:
                # shows the congrats message instantly
                st.balloons()
                # message(congrats_msg, is_user=False)

                # update state for next rendering
                image_clustered = Image.fromarray(
                    st.session_state.session_tabs_state[tab_idx]['image_clustered_flattened'].reshape(orig_shape))
                st.session_state.session_tabs_state[tab_idx]['in_progress'] = False
                st.session_state.session_tabs_state[tab_idx].pop(
                    'image_to_color_flattened')
                st.session_state.session_tabs_state[tab_idx]['messages'].append(
                    {'role': 'image', 'content': (image, image_clustered)})
                st.session_state.session_tabs_state[tab_idx]['messages'].append(
                    {'role': 'assistant', 'content': congrats_msg})
                st.experimental_rerun()

        elif st.session_state[f'{tab_idx}_coords']:
            # a coordinate in the image has been clicked
            color_region(st.session_state[f'{tab_idx}_coords'], tab_idx, color)


# this is setup at app launch
if 'session_tabs_state' not in st.session_state:
    st.session_state.session_tabs_state = []

st.set_page_config(
    page_title="Arther",
    page_icon="👋",
)
congrats_msg = "Congratulations! You have completed an Arther paint session. \
    On the left is the sketch you painted, and on the right is the original painting. \
    You can start another session by clicking 'Start session' from the sidebar."


st.sidebar.header('Arther')
st.sidebar.write(
    'Arther is an AI-powered paint-by-click app that generates a thematic sketch and segments it into regions \
    of identical colors. The user can then fill each region with their selected colors. Upon completion Arther \
    also reveals the original colors as produced by the generative AI.')

st.sidebar.button('Start session', key='start_session',
                  type="primary", on_click=start_session)

themes = ['Nature and Landscapes', 'Mandalas and Geometric Patterns',
          'Fantasy and Mythology', 'Animals and Wildlife', 'Seasons and Holidays', 'Abstract and Psychedelic Art',
          'Artistic Styles and Art History', 'Cultural and Ethnic Art']

message("Welcome to Arther! Your own ai-powered personalized paint book!")
message("To start a paint session, press the \'Start session\' button in the sidebar.")

session_tab_names = [
    "session " + str(i+1) for i in range(len(st.session_state.session_tabs_state))]
session_tab_names.reverse()
session_tabs = []
if len(session_tab_names) > 0:
    session_tabs = st.tabs(session_tab_names)

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

for i, t in enumerate(reversed(session_tabs)):
    messages = st.session_state.session_tabs_state[i]['messages']
    last_selected_theme_idx = st.session_state.session_tabs_state[i]['last_selected_theme_idx']

    # render messages in order
    for j, m in enumerate(messages):
        with t:
            if m['role'] == 'image':
                col1, col2 = st.columns(2)
                col1.image(m['content'][0])
                col2.image(m['content'][1])
            elif m['role'] == 'theme_options':
                option_menu(None, themes, on_change=theme_callback,
                            key=f'{i}_theme_options', default_index=last_selected_theme_idx)
            elif m['role'] == 'prompt_options':
                k = m['content']
                op_list = st.session_state.session_tabs_state[i]['prompt_options']
                option_menu(None, op_list[k][1], on_change=prompt_callback,
                            key=f'{i}_{k}_prompt_options', default_index=op_list[k][0])
            else:
                message(m['content'], is_user=True if m['role'] ==
                        'user' else False, key=f'{i}_{j}_chat')

    # render in-progress image
    render_image(t, i)
