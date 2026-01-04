import pandas as pd
import numpy as np
import torch
from PIL import Image
import faiss
from faiss import write_index, read_index
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
import yaml
from config import ProjectArgs

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


logger: logging.Logger = logging.getLogger(__name__)

_args = ProjectArgs().value


def create_dataframe(root_path):
    """
    Walks through the directory structure and creates a Pandas DataFrame
    containing paths to all image files.

    Args:
        root_path (str): The root directory to search for images.

    Returns:
        pandas.DataFrame: A DataFrame with image paths.
    """
    try:
        images_list = list()

        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith((".png", "jpg", "jpeg")):
                    images_list.append(os.path.join(root, file))
        df = pd.DataFrame(images_list, columns=["image_path"])
        return df
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        return None


def load_image(image_path):
    """
    Loads an image from a given path and converts it to RGB format.

    Args:
        image_path (str): The path to the image file.

    Returns:
        PIL.Image.Image: The loaded image object.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def extract_features(image, image_processor, model):
    """
    Extracts embeddings from an image using a PyTorch model.

    Args:
        image (PIL.Image.Image): The input image.
        image_processor (object): An object with a '__call__' method for processing images.
        model (torch.nn.Module): The PyTorch model for feature extraction.

    Returns:
        numpy.ndarray: The extracted feature vector.
    """
    try:
        image = load_image(image)
        inputs = image_processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state
        embedding = embedding[:, 0, :].squeeze(1)
        return embedding.numpy()
    except Exception as e:
        logger.error(f"Error extracting features from {image}: {e}")
        return None


def create_batch(dataframe, image_processor, model):
    """
    Extracts embeddings for all images in the DataFrame.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing image paths.
        image_processor (object): An object with a '__call__' method for processing images.
        model (torch.nn.Module): The PyTorch model for feature extraction.

    Returns:
        numpy.ndarray: A NumPy array containing the extracted embeddings.
    """
    try:
        return np.vstack(
            [
                extract_features(path, image_processor, model)
                for path in dataframe["image_path"]
            ]
        )
    except Exception as e:
        logger.error(f"Creating batch failed: {e}")


def create_index(embeddings):
    """
    Creates a Faiss index from the extracted embeddings.

    Args:
        embeddings (numpy.ndarray): The embeddings to index.

    Returns:
        faiss.Index: The Faiss index.
    """
    vector_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(embeddings)
    logger.info(f"Index created.")
    return index


def save_index(index, df):
    """
    Saves the Faiss index to a file.

    Args:
        index (faiss.Index): The Faiss index to save.
        index_path (str): The path to save the index file.
    """
    try:
        df.to_csv(_args.index_path + datetime.now().strftime("%d-%m-%y_%H-%M") + ".csv")
        write_index(
            index,
            _args.index_path + datetime.now().strftime("%d-%m%y_%H-%M") + ".index",
        )
    except Exception as e:
        logger.error(f"Failed saving index: {e}")


def load_index(index_path):
    """
    Loads a Faiss index from a file.

    Args:
        index_path (str): The path to the index file.

    Returns:
        faiss.Index: The loaded Faiss index.
    """
    try:
        return read_index(index_path)
    except Exception as e:
        logger.error(f"Error loading Faiss index: {e}")
        return None


def load_df(path):
    """
    Loads a Pandas DataFrame from a CSV file.
    """
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {path}: {e}")
        return None
