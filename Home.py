import streamlit as st
import numpy as np
import os
from PIL import Image
from PIL import ImageColor
from sklearn.cluster import KMeans
import random
import cv2
from streamlit_chat import message
import openai
from streamlit_option_menu import option_menu
from streamlit_image_coordinates import streamlit_image_coordinates
import random
from langchain.llms import OpenAI
from stability_sdk.api import Context, generation
import time
from st_click_detector import click_detector


stability_engine_id = "stable-diffusion-xl-beta-v2-2-2"
stability_api_host = os.getenv('API_HOST', 'https://api.stability.ai')
stability_api_key = os.getenv("STABILITY_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model_id = os.getenv("OPENAI_MODEL_ID")

stability_api = Context(host='grpc.stability.ai:443', api_key=stability_api_key,
                        generate_engine_id=stability_engine_id)

congrats_msg = "Congratulations! You have completed an Arther coloring session. \
    On the left is the sketch you colored, and on the right is the original drawing. \
    You can start another session by clicking 'Start session' from the sidebar."

themes = ['Nature and Landscapes', 'Mandalas and Geometric Patterns',
          'Fantasy and Mythology', 'Animals and Wildlife', 'Seasons and Holidays', 'Abstract and Psychedelic Art',
          'Artistic Styles and Art History', 'Cultural and Ethnic Art']

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


class SessionTab:
    def __init__(self):
        self.messages = [{'role': 'assistant', 'content': 'For your new session I have a few suggestions for a theme that you can explore.'}, {
            'role': 'theme_options'}]
        self.in_progress = True
        self.last_selected_theme_idx = 0
        self.prompt_options = []
        self.image_to_color_flattened = None
        self.image_clustered_flattened = None
        self.orig_shape = None
        self.color_labels = None
        self.color_clusters = None
        self.cluster_centers = None
        self.contours = None
        self.colored_regions = []
        self.last_was_undo = False


@st.cache_data(show_spinner='')
def auto_canny(image, sigma=0.33):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower, upper)

    # inversion
    edges = 255 - edges

    # to increase contrast
    edges = cv2.createCLAHE().apply(edges)

    # gaussian blur to remove aliasing
    smoothed_edges = cv2.GaussianBlur(edges, (3, 3), 0)
    return smoothed_edges


@st.cache_data(show_spinner="Generating drawing...")
def getImageFromText(prompt: str) -> Image.Image:
    results = stability_api.generate(prompts=[f'a brightly colored drawing with big shapes showing {prompt}', 'person making a drawing'], weights=[
        1, -50], width=512, height=512, steps=30)
    return results[generation.ARTIFACT_IMAGE][0]

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
    llm = OpenAI(model_name=openai_model_id, max_tokens=25,
                 n=3, best_of=3, presence_penalty=1.0)
    with st.spinner(''):
        llm_result = llm.generate([prompt.lower()+" ->"])

    comps = []
    # generation[0] is the list of generations for the first entry in the input list
    for g in llm_result.generations[0]:
        comps.append(g.text.split(".")[0] + ".")
    return comps

    ''' 
    with st.spinner(''):
        completionResponse = openai.Completion.create(
            model=openai_model_id, prompt=prompt.lower()+" ->", max_tokens=30, n=3, best_of=3, presence_penalty=1.0)

    comps = []
    for c in completionResponse.choices:
        comps.append(c.text.split(".")[0] + ".")
    return comps
    '''

    '''
    response = requests.post('http://localhost:8080/cluster-image',json={'image': to_be_clustered.tolist()}).json()
    labels = np.array(response['labels'])
    clusters = {int(k): v for k, v in response['clusters'].items()}
    cluster_centers = np.array(response['cluster_centers'])
    clustered_flat = np.array(response['clustered_image_flat'], dtype='uint8')
    '''


def load_image(prompt: str, session_tab_idx: int):
    image = getImageFromText(prompt)
    to_be_clustered = np.array(image)

    labels, clusters, cluster_centers, clustered_flat = getKMeans(to_be_clustered, 10)

    gs_image = np.array(image.convert('L'))
    image_to_color = auto_canny(gs_image)
    image_to_color = cv2.cvtColor(
        image_to_color, cv2.COLOR_GRAY2RGB).reshape((-1, 3))

    # contours, _ = cv2.findContours(255-image_to_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    st.session_state.session_tabs[session_tab_idx].image_to_color_orig = image_to_color
    st.session_state.session_tabs[session_tab_idx].image_to_color_flattened = image_to_color.copy(
    )
    st.session_state.session_tabs[session_tab_idx].image_clustered_flattened = clustered_flat
    st.session_state.session_tabs[session_tab_idx].orig_shape = to_be_clustered.shape
    st.session_state.session_tabs[session_tab_idx].color_labels = labels
    st.session_state.session_tabs[session_tab_idx].color_clusters = clusters
    st.session_state.session_tabs[session_tab_idx].cluster_centers = cluster_centers
    st.session_state.session_tabs[session_tab_idx].cluster_colors_hex = [
        '#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2]) for c in cluster_centers]

    st.session_state[f'{session_tab_idx}_coords'] = None


def prompt_callback(key: str):
    tab_idx = int(key.split("_")[0])
    if not st.session_state.session_tabs[tab_idx].in_progress:
        st.error('This session is already complete!')
        return

    prompt_set_idx = int(key.split("_")[1])

    prompt_options = st.session_state.session_tabs[tab_idx].prompt_options[prompt_set_idx][1]
    last_selected_prompt_idx = 0
    for i, p in enumerate(prompt_options):
        if p == st.session_state[key]:
            last_selected_prompt_idx = i
            break
    st.session_state.session_tabs[tab_idx].prompt_options[prompt_set_idx] = (
        last_selected_prompt_idx, prompt_options)
    st.session_state.session_tabs[tab_idx].messages.append(
        {'role': 'user', 'content': st.session_state[key]})
    st.session_state.session_tabs[tab_idx].messages.append(
        {'role': 'assistant', 'content': 'To color the sketch, select a color from the color-picker just below the image, and click on an uncolored \
            region in the sketch. This will fill all the regions that share the color with the clicked point.'})
    load_image(st.session_state[key], tab_idx)


def theme_callback(key: str):
    tab_idx = int(key.split("_")[0])
    if not st.session_state.session_tabs[tab_idx].in_progress:
        # the session is already complete
        st.info('This session is already complete!')
        return

    last_selected_theme_idx = 0
    for i, t in enumerate(themes):
        if t == st.session_state[key]:
            last_selected_theme_idx = i
            break
    st.session_state.session_tabs[tab_idx].last_selected_theme_idx = last_selected_theme_idx

    st.session_state.session_tabs[tab_idx].prompt_options.append(
        (0, get_completions(st.session_state[key])))

    st.session_state.session_tabs[tab_idx].messages.append(
        {"role": "user", "content": st.session_state[key]})
    st.session_state.session_tabs[tab_idx].messages.append(
        {"role": "assistant", "content": "Here are a few suggestions for the contents of your sketch."})

    # for prompt options, include the index of the most recently added
    st.session_state.session_tabs[tab_idx].messages.append(
        {"role": "prompt_options", "content": len(st.session_state.session_tabs[tab_idx].prompt_options)-1})


def start_session():
    st.session_state.session_tabs.append(SessionTab())


def uncolor_last_region(tab_idx):
    if len(st.session_state.session_tabs[tab_idx].colored_regions) == 0:
        return
    image_to_color_flattened = st.session_state.session_tabs[tab_idx].image_to_color_flattened
    image_to_color_orig = st.session_state.session_tabs[tab_idx].image_to_color_orig
    color_clusters = st.session_state.session_tabs[tab_idx].color_clusters
    last_colored_region = st.session_state.session_tabs[tab_idx].colored_regions.pop(
    )
    pixels = color_clusters[last_colored_region]
    for pIdx in pixels:
        image_to_color_flattened[pIdx] = image_to_color_orig[pIdx]

    # before we rerun, we need to disable the selected coords so that color_region() won't
    # run again, but unfortunately `st.session_state[f'{tab_idx}_coords'] = None` doesn't work
    st.session_state.session_tabs[tab_idx].last_was_undo = True
    st.experimental_rerun()


def color_region(coords, tab_idx, hex_color):
    color_clusters = st.session_state.session_tabs[tab_idx].color_clusters
    color_labels = st.session_state.session_tabs[tab_idx].color_labels
    image_to_color_flattened = st.session_state.session_tabs[tab_idx].image_to_color_flattened
    idx = coords['y'] * \
        st.session_state.session_tabs[tab_idx].orig_shape[0] + coords['x']

    if color_labels[idx] in st.session_state.session_tabs[tab_idx].colored_regions:
        return
    pixels = color_clusters[color_labels[idx]]
    for pIdx in pixels:
        image_to_color_flattened[pIdx] = list(
            ImageColor.getcolor(hex_color, 'RGB'))
    st.session_state.session_tabs[tab_idx].colored_regions.append(
        color_labels[idx])

    st.experimental_rerun()


def render_image(tab, tab_idx: int):
    if st.session_state.session_tabs[tab_idx].image_to_color_flattened is None:
        return

    color_selector_prefix = """
        <div style='display: flex; flex-direction: column; align-items: center;'>
            <div style='display: flex; justify-content: start;'> \n"""
    color_selector_suffix = """</div>\n</div>"""
    with tab:
        image_to_color_flattened = st.session_state.session_tabs[tab_idx].image_to_color_flattened
        orig_shape = st.session_state.session_tabs[tab_idx].orig_shape
        num_colored_regions = len(
            st.session_state.session_tabs[tab_idx].colored_regions)
        num_color_clusters = len(
            st.session_state.session_tabs[tab_idx].color_clusters)

        image_to_color = image_to_color_flattened.reshape(orig_shape)
        image = Image.fromarray(image_to_color)
        streamlit_image_coordinates(image, key=f'{tab_idx}_coords')
        st.components.v1.html(imageCursorJs)

        col1, col2 = st.columns(2)
        with col1:
            for c in st.session_state.session_tabs[tab_idx].cluster_colors_hex:
                color_selector_prefix += f"<a href='#' id='{c}' style='background-color: {c}; width: 25px; height: 25px; margin: 5px; display: flex; flex-wrap: wrap; border-radius: 5px; justify-content: center; align-items: stretch;'></a>\n"
            color = click_detector(color_selector_prefix+color_selector_suffix)
        with col2:
            st.button('Undo last stroke', key=f'undo_{tab_idx}')
            if st.session_state[f'undo_{tab_idx}'] and st.session_state.session_tabs[tab_idx].in_progress:
                uncolor_last_region(tab_idx)

        col_progress = int(100*num_colored_regions/num_color_clusters)
        st.progress(col_progress, text=f'{col_progress}% completed')
        if col_progress >= 100:
            st.balloons()
            time.sleep(2)
            if st.session_state.session_tabs[tab_idx].in_progress:
                # update state for next rendering
                st.session_state.session_tabs[tab_idx].in_progress = False
                image_clustered = Image.fromarray(
                    st.session_state.session_tabs[tab_idx].image_clustered_flattened.reshape(orig_shape))
                st.session_state.session_tabs[tab_idx].image_to_color_flattened = None
                st.session_state.session_tabs[tab_idx].messages.append(
                    {'role': 'image', 'content': (image, image_clustered)})
                st.session_state.session_tabs[tab_idx].messages.append(
                    {'role': 'assistant', 'content': congrats_msg})

                st.experimental_rerun()

        elif st.session_state[f'{tab_idx}_coords']:
            if st.session_state.session_tabs[tab_idx].last_was_undo:
                st.session_state.session_tabs[tab_idx].last_was_undo = False
            else:
                if not color:
                    st.warning('Please select a color.')
                    st.stop()
                color_region(
                    st.session_state[f'{tab_idx}_coords'], tab_idx, color)


def render_messages(tab, session_tab_idx: int):
    for j, m in enumerate(st.session_state.session_tabs[session_tab_idx].messages):
        with tab:
            if m['role'] == 'image':
                col1, col2 = st.columns(2)
                col1.image(m['content'][0])
                col2.image(m['content'][1])
            elif m['role'] == 'theme_options':
                option_menu(None, themes, on_change=theme_callback,
                            key=f'{session_tab_idx}_theme_options', default_index=st.session_state.session_tabs[session_tab_idx].last_selected_theme_idx)
            elif m['role'] == 'prompt_options':
                k = m['content']
                def_idx = st.session_state.session_tabs[session_tab_idx].prompt_options[k][0]
                op_list = st.session_state.session_tabs[session_tab_idx].prompt_options[k][1]
                option_menu(None, op_list, on_change=prompt_callback,
                            key=f'{session_tab_idx}_{k}_prompt_options', default_index=def_idx)
            else:
                message(m['content'], is_user=True if m['role'] ==
                        'user' else False, key=f'{session_tab_idx}_{j}_chat')


def main():

    # this is setup at app launch
    if 'session_tabs' not in st.session_state:
        st.session_state.session_tabs = []

    st.set_page_config(page_title="Arther", page_icon="👋")
    st.sidebar.header('Arther')
    st.sidebar.write(
        'Arther is an AI-powered color-by-click app that generates a thematic sketch and segments it into regions \
    of identical colors. The user can then fill each region with their selected colors. Upon completion Arther \
    also reveals the original colors as produced by the generative AI.')

    st.sidebar.button('Start session', key='start_session',
                      type="primary", on_click=start_session)

    message("Welcome to Arther! Your own AI-powered personalized coloring book!")
    message(
        "To start a coloring session, press the \'Start session\' button in the sidebar.")

    session_tab_names = [
        "session " + str(i+1) for i in range(len(st.session_state.session_tabs))]
    session_tab_names.reverse()
    session_tabs = []
    if len(session_tab_names) > 0:
        session_tabs = st.tabs(session_tab_names)

    for i, t in enumerate(reversed(session_tabs)):
        render_messages(t, i)
        render_image(t, i)


if __name__ == "__main__":
    main()
