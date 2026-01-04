from itertools import cycle

import streamlit as st
from transformers import AutoImageProcessor, AutoModel
from utils import (
    create_batch,
    create_dataframe,
    create_index,
    extract_features,
    load_df,
    load_index,
    save_index,
)

from .utils._logger import logger
from .utils._validation import config_args

# States
if "search_status" not in st.session_state:
    st.session_state["search_status"] = False

if "can_search" not in st.session_state:
    st.session_state["can_search"] = False

if "dataframe" not in st.session_state:
    st.session_state["dataframe"] = None

if "index" not in st.session_state:
    st.session_state["index"] = None


class ImageSearchApp:
    """
    A Streamlit application for image searching using a pre-trained model.
    """

    def __init__(self):
        self.session_state = st.session_state  # Access session state directly
        self.image_processor = None
        self.model = None
        self.search_path = None

    def load_model(self):
        """Loads the pre-trained image processor and model."""
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                config_args.model_path, local_files_only=True, use_fast=True
            )
            self.model = AutoModel.from_pretrained(
                config_args.model_path, local_files_only=True
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_data(self, search_path):
        """Loads the dataframe and index from the specified path."""
        try:
            # st.session_state.dataframe = create_dataframe(search_path)
            st.session_state.dataframe = create_dataframe(search_path)
            embeddings = create_batch(
                st.session_state.dataframe, self.image_processor, self.model
            )
            st.session_state.index = create_index(embeddings)
            save_index(st.session_state.index, st.session_state.dataframe)
            logger.info(f"Data loaded successfully from {search_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def search_image(self, image_path):
        """Searches for similar images."""
        try:
            embeds = extract_features(image_path, self.image_processor, self.model)
            D, I = st.session_state.index.search(embeds, k=12)
            cols = cycle(st.columns(4))
            for images in st.session_state.dataframe.loc[I[0], "image_path"].values:
                next(cols).image(images, use_container_width=True)
            st.sidebar.image(self.search_path)
            logger.info(f"Search completed for {image_path}")
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def run(self):
        """Runs the Streamlit application."""

        # Streamlit app title
        st.set_page_config(page_title="Pixel Match")

        self.load_model()

        # Search interface
        search_path = st.sidebar.text_input(
            "Search folder path",
            help="Initilize the first search for creation of index and dataframe",
        ).replace('"', "")

        if st.session_state.search_status == False:
            with st.sidebar:
                with st.spinner():
                    if search_path:
                        st.session_state.search_status = True
                        st.session_state.can_search = True
                        self.search_path = search_path
                        self.load_data(search_path)
                        st.sidebar.success(
                            f"{len(st.session_state.dataframe)} images uploaded."
                        )
                        logger.info(
                            f"{len(st.session_state.dataframe)} images uploaded."
                        )
        load_form = st.sidebar.form("load")
        loaded_index_path: str = load_form.text_input("Index path").replace('"', "")
        loaded_df_path: str = load_form.text_input("Dataframe path").replace('"', "")
        submitted: bool = load_form.form_submit_button("Submit")
        if submitted:
            with st.spinner():
                st.session_state.index = load_index(loaded_index_path)
                st.session_state.dataframe = load_df(loaded_df_path)
                st.sidebar.success(f"{len(st.session_state.dataframe)} images loaded.")
                logger.info(f"{len(st.session_state.dataframe)} images loaded.")
                st.session_state.can_search = True

        if st.session_state.can_search == True:
            search_image_path = st.file_uploader(
                "Choose an image to search", type=["png", "jpg", "jpeg"]
            )
            if search_image_path:
                st.divider()
                self.search_path = search_image_path
                self.search_image(search_image_path)


app = ImageSearchApp()
app.run()
