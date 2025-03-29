# built-in dependencies
import time
import logging.config
from typing import Any, Dict, Union, List, Tuple

# 3rd party dependencies
import cupy as np

# project dependencies
import ThirdParty.deepface.deepface as deepface
from ThirdParty.deepface.deepface.models.FacialRecognition import FacialRecognition

logger = logging.getLogger(__name__)


def verify(
    img1_path: Union[str, np.ndarray, List[float]],
    img2_path: Union[str, np.ndarray, List[float]],
    model_name: str = "GhostFaceNet",
    distance_metric: str = "euclidean",
) -> Dict[str, Any]:
    """
    Verify if an image pair represents the same person or different persons.

    The verification function converts facial images to vectors and calculates the similarity
    between those vectors. Vectors of images of the same person should exhibit higher similarity
    (or lower distance) than vectors of images of different persons.

    Args:
        img1_path (str or np.ndarray or List[float]): Path to the first image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images
            or pre-calculated embeddings.

        img2_path (str or np.ndarray or List[float]): Path to the second image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images
            or pre-calculated embeddings.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).


        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)

    Returns:
        result (dict): A dictionary containing verification results.

        - 'verified' (bool): Indicates whether the images represent the same person (True)
            or different persons (False).

        - 'distance' (float): The distance measure between the face vectors.
            A lower distance indicates higher similarity.

        - 'max_threshold_to_verify' (float): The maximum threshold used for verification.
            If the distance is below this threshold, the images are considered a match.

        - 'model' (str): The chosen face recognition model.

        - 'similarity_metric' (str): The chosen similarity metric for measuring distances.

        - 'time' (float): Time taken for the verification process in seconds.
    """

    tic = time.time()

    # model: FacialRecognition = modeling.build_model(model_name)
    dims = 512  # model.output_shape

    # extract faces from img1
    if isinstance(img1_path, list):
        # given image is already pre-calculated embedding
        if not all(isinstance(dim, float) for dim in img1_path):
            raise ValueError(
                "When passing img1_path as a list, ensure that all its items are of type float."
            )

        if len(img1_path) != dims:
            raise ValueError(
                f"embeddings of {model_name} should have {dims} dimensions,"
                f" but it has {len(img1_path)} dimensions input"
            )

        img1_embeddings = [img1_path]
        img1_facial_areas = [None]

    # extract faces from img2
    if isinstance(img2_path, list):
        # given image is already pre-calculated embedding
        if not all(isinstance(dim, float) for dim in img2_path):
            raise ValueError(
                "When passing img2_path as a list, ensure that all its items are of type float."
            )

        if len(img2_path) != dims:
            raise ValueError(
                f"embeddings of {model_name} should have {dims} dimensions,"
                f" but it has {len(img2_path)} dimensions input"
            )

        img2_embeddings = [img2_path]
        img2_facial_areas = [None]

    distances = []
    facial_areas = []
    for idx, img1_embedding in enumerate(img1_embeddings):
        # logger.info("img1_embeddings: {} - idx: {}".format(img1_embedding, idx))
        for idy, img2_embedding in enumerate(img2_embeddings):
            # logger.info("img2_embeddings: {} - idy: {}".format(img2_embedding, idy))
            distance = find_distance(img1_embedding, img2_embedding, distance_metric)
            # logger.info("distance: {}".format(distance))
            distances.append(distance)

    # find the face pair with minimum distance
    threshold = find_threshold(model_name, distance_metric)
    # logger.info("threshold: {}".format(threshold))
    distance = float(min(distances))  # best distance
    # logger.info("distance: {}".format(distance))

    toc = time.time()

    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "similarity_metric": distance_metric,
        "time": round(toc - tic, 2),
    }

    return resp_obj


def find_cosine_distance_v1(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find cosine distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def find_cosine_distance_v3(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find cosine distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    return None


def find_cosine_distance_v2(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find cosine distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    return np.dot(source_representation, test_representation) / (np.linalg.norm(source_representation) * np.linalg.norm(test_representation))


def find_euclidean_distance_v1(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find euclidean distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated euclidean distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def find_euclidean_distance_v2(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find euclidean distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated euclidean distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    return np.linalg.norm(source_representation - test_representation)


def find_euclidean_distance_v3(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find euclidean distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated euclidean distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    return None


def l2_normalize(x: Union[np.ndarray, list]) -> np.ndarray:
    """
    Normalize input vector with l2
    Args:
        x (np.ndarray or list): given vector
    Returns:
        y (np.ndarray): l2 normalized vector
    """
    if isinstance(x, list):
        x = np.array(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def find_distance(
    alpha_embedding: Union[np.ndarray, list],
    beta_embedding: Union[np.ndarray, list],
    distance_metric: str,
) -> np.float64:
    """
    Wrapper to find distance between vectors according to the given distance metric
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if distance_metric == "cosine_v1":
        distance = find_cosine_distance_v1(alpha_embedding, beta_embedding)
    elif distance_metric == "cosine_v2":
        distance = find_cosine_distance_v2(alpha_embedding, beta_embedding)
    elif distance_metric == "cosine_v3":
        distance = find_cosine_distance_v3(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_v1":
        distance = find_euclidean_distance_v1(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_v2":
        distance = find_euclidean_distance_v2(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_v3":
        distance = find_euclidean_distance_v3(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        distance = find_euclidean_distance_v1(
            l2_normalize(alpha_embedding), l2_normalize(beta_embedding)
        )
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return distance


def find_threshold(model_name: str, distance_metric: str) -> float:
    """
    Retrieve pre-tuned threshold values for a model and distance metric pair
    Args:
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        distance_metric (str): distance metric name. Options are cosine, euclidean
            and euclidean_l2.
    Returns:
        threshold (float): threshold value for that model name and distance metric
            pair. Distances less than this threshold will be classified same person.
    """
    if distance_metric == "euclidean_v1" or distance_metric == "euclidean_v2":
        distance_metric = "euclidean"
    if distance_metric == "cosine_v1" or distance_metric == "cosine_v2":
        distance_metric = "cosine"
    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

    thresholds = {
        # "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}, # 2622d
        "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17},  # 4096d - tuned with LFW
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
        "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "euclidean_l2": 1.10},
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

    return threshold
