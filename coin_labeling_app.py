import json

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

NAMES = ['Paul', 'Marine', 'Seb']

def init_app(reset_train_labels=False):
    if 'human' not in st.session_state:
        st.session_state.human = NAMES[0]

    if 'train_labels' not in st.session_state or reset_train_labels:
        labels = pd.read_csv('project/data/train_labels.csv')
        n_labels = len(labels)
        if st.session_state.human == NAMES[0]:
            labels = labels.iloc[:n_labels//3].copy()
        elif st.session_state.human == NAMES[1]:
            labels = labels.iloc[n_labels//3:2*n_labels//3].copy()
        else:
            labels = labels.iloc[2*n_labels//3:].copy()
        st.session_state.train_labels = labels
    if 'labels' not in st.session_state:
        st.session_state.labels = list(st.session_state.train_labels.columns[1:]) + ["Not a coin"]

    if reset_train_labels:
        if 'coin_labels' in st.session_state:
            st.session_state.__delitem__('coin_labels')



def build_app():
    # Set the title of the web app
    st.set_page_config(layout="wide")
    st.title('Coin Labeling App')

    with st.sidebar:
        st.session_state.human = st.selectbox('Human', NAMES)
        init_app()
        if st.button('Reset app data'):
            init_app(reset_train_labels=True)

    # Load the data
    st.subheader('Train labels')
    train_labels = st.session_state.train_labels
    st.dataframe(train_labels, use_container_width=True)

    if st.button('Load labelled data'):
        with open(f'project/data/coin_train_labels_{st.session_state.human}.json', 'r') as f:
            st.session_state.coin_labels = json.load(f)
    if 'coin_labels' in st.session_state:
        with st.sidebar:
            index = st.number_input('Index', min_value=train_labels.index[0], max_value=train_labels.index[-1], value=train_labels.index[0])
            data = train_labels.loc[index]
            # Show data that is not zero
            st.divider()
            st.dataframe(data[data != 0].to_frame(), use_container_width=True)

        # Show the image
        subdirs_names = os.listdir('project/data/train')
        possible_hole_image_path = [f'project/data/train/{s}/{data["id"]}.JPG' for s in subdirs_names]
        filtered_hole_image_path = [path for path in possible_hole_image_path if os.path.exists(path)]
        if len(filtered_hole_image_path) > 0:
            hole_image_path = filtered_hole_image_path[0]
        else:
            hole_image_path = "No image found"
            st.error('No image found')

        subdirs_names = os.listdir('project/data/train_croped_coins')
        possible_image_dir = [f'project/data/train_croped_coins/{s}/{data["id"]}/'for s in subdirs_names]
        filter_image_dir = [path for path in possible_image_dir if os.path.exists(path)]

        if len(filter_image_dir) > 0:
            coin_images_dir = filter_image_dir[0]
        else:
            coin_images_dir = "No image found"
            st.error('No image found')
        st.subheader(f"Labeling image: {data['id']}  | path: {hole_image_path}")
        with st.expander('Hole image'):
            st.image(hole_image_path, width=500)
        st.divider()

        # coin_image_paths = [f"{coin_images_dir}{data['id']}_{i}.JPG" for i in range(3)]
        # coin_image_paths = [path for path in coin_image_paths if os.path.exists(path)]
        cols = st.columns(2)
        cols[0].dataframe(data[data != 0].to_frame().T, use_container_width=True)
        width = cols[1].number_input('Image width', min_value=100, max_value=500, value=200)

        coin_image_paths = os.listdir(coin_images_dir)
        coin_image_paths = [os.path.join(coin_images_dir, path) for path in coin_image_paths if path.endswith('.JPG')]

        max_n_coins = 8
        cols = st.columns(len(coin_image_paths[:max_n_coins]))
        labels = [None for _ in coin_image_paths]
        for i, (col, p) in enumerate(zip(cols, coin_image_paths)):
            label = st.session_state.coin_labels.get(p, None)
            if label is not None:
                col.warning('Label already set')
                col.text(p)
            label_index = st.session_state.labels.index(label) if label is not None else 0
            labels[i] = col.selectbox('Label', st.session_state.labels, key=f'label{i}', index=label_index)
            col.image(p, width=width)

        cols = st.columns(len(coin_image_paths[max_n_coins:])) if len(coin_image_paths) > max_n_coins else []
        for j, (col, p) in enumerate(zip(cols, coin_image_paths[max_n_coins:])):
            i = j + max_n_coins
            label = st.session_state.coin_labels.get(p, None)
            if label is not None:
                col.warning('Label already set')
                col.text(p)
            col.text(p)
            label_index = st.session_state.labels.index(label) if label is not None else 0
            labels[i] = col.selectbox('Label', st.session_state.labels, key=f'label{i}', index=label_index)
            col.image(p, width=width)

        if st.button('Save labels'):
            for i, (p, label) in enumerate(zip(coin_image_paths, labels)):
                if label is not None:
                    st.session_state.coin_labels[p] = label
            with open(f'project/data/coin_train_labels_{st.session_state.human}.json', 'w') as f:
                json.dump(st.session_state.coin_labels, f, indent=4)
            st.success('Labels saved successfully')

        if st.toggle('Show all labels', value=True):
            st.write(st.session_state.coin_labels)


if __name__ == '__main__':
    build_app()
