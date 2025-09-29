from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from blib2to3.pgen2.driver import Iterable
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import MiniBatchKMeans

from config import (
    SIZE,
    MIN_TILE_SIZE,
    MAX_TILE_SIZE,
    BORDER_COLOR,
    CLUSTERS,
    COLOR_DISTANCE_THRESHOLD,
    COLOR_FREQUENCY_THRESHOLD,
)


# Largely stolen from Towards Data Science
# https://towardsdatascience.com/colour-image-quantization-using-k-means-636d93887061/
def quantize_colors(image, clusters) -> np.ndarray:
    """
    Apply color quantization to the image to be turned into a mosaic.
    Args:
        image (np.ndarray): Image to be quantized.
        clusters (int): Number of colors you wish to use :)

    Returns: color quantized image in RGB colorspace

    """
    height, width = image.shape[:2]
    # for better...eye similarity
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # using MiniBatchKMeans for speed
    model = MiniBatchKMeans(n_clusters=clusters)
    labels = model.fit_predict(image)
    quantized = model.cluster_centers_.astype("uint8")[labels]
    quantized_image = quantized.reshape((height, width, 3))
    return cv2.cvtColor(quantized_image, cv2.COLOR_LAB2RGB)


def add_tile_border(tile: np.ndarray) -> np.ndarray:
    tile[0, :] = BORDER_COLOR
    tile[-1, :] = BORDER_COLOR
    tile[:, 0] = BORDER_COLOR
    tile[:, -1] = BORDER_COLOR
    return tile


def color_difference(color1, color2) -> float:
    diff = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    return diff


def create_tile(image: np.ndarray, min_size) -> np.ndarray:
    height, width = image.shape[:2]
    pixels = image.reshape(-1, 3)
    pixels = [tuple(pixel) for pixel in pixels]
    color_counts = Counter(pixels)

    try:
        most_common_color = color_counts.most_common(1)[0][0]
        most_common_color_occurrence = color_counts.most_common(1)[0][1]
    except:
        return image

    small = height <= min_size or width <= min_size
    color_rel_freq = most_common_color_occurrence / len(pixels)
    if color_rel_freq >= COLOR_FREQUENCY_THRESHOLD or small:
        # color_rel_freq will be one if there is only one color
        recolored = np.full((height * width, 3), most_common_color, dtype=np.uint8)
        shaped = recolored.reshape((height, width, 3))
        return add_tile_border(shaped)
    else:
        if len(color_counts) >= 2:
            second_most_common_color = color_counts.most_common(2)[1][0]

            if (
                color_difference(most_common_color, second_most_common_color)
                > COLOR_DISTANCE_THRESHOLD
            ):
                mid_h = height // 2
                mid_w = width // 2

                # Fixed quadrant slicing
                quad1 = create_tile(image[0:mid_h, 0:mid_w], min_size)
                quad2 = create_tile(image[0:mid_h, mid_w:width], min_size)
                quad3 = create_tile(image[mid_h:height, 0:mid_w], min_size)
                quad4 = create_tile(image[mid_h:height, mid_w:width], min_size)

                # Fixed concatenation
                top_side = np.concatenate([quad1, quad2], axis=1)
                bottom_side = np.concatenate([quad3, quad4], axis=1)
                tile = np.concatenate([top_side, bottom_side], axis=0)

                return tile
            else:
                recolored = np.full(
                    (height * width, 3), most_common_color, dtype=np.uint8
                )
                shaped = recolored.reshape((height, width, 3))
                return add_tile_border(shaped)
        else:
            recolored = np.full((height * width, 3), most_common_color, dtype=np.uint8)
            shaped = recolored.reshape((height, width, 3))
            return add_tile_border(shaped)


def tile_image(
    image: np.ndarray, max_size=MAX_TILE_SIZE, min_size=MIN_TILE_SIZE
) -> np.ndarray:
    height, width = image.shape[:2]
    r_start = 0
    c_start = 0
    r_end = max_size
    c_end = max_size
    while r_end <= height and r_start <= height:
        image[r_start:r_end, c_start:c_end] = create_tile(
            image[r_start:r_end, c_start:c_end], min_size
        )
        c_start += max_size
        c_end += max_size
        c_end = min(width, c_end)
        if c_end > width or c_start >= c_end:
            r_start += max_size
            r_end += max_size
            r_end = min(r_end, height)
            c_start = 0
            c_end = max_size
    return image


def resize_image(image, size):
    # resize image to fixed(ish) size (because aspect ratios might be different)
    # smaller dimension will be size specified above
    width = image.shape[0]
    height = image.shape[1]
    min_dim = min(width, height)
    scaling_factor = size / min_dim
    # Using different interpolation methods for upscaling vs downscaling
    if scaling_factor < 1:
        resized_image = cv2.resize(
            image,
            (0, 0),
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized_image = cv2.resize(
            image,
            (0, 0),
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_CUBIC,
        )
    return resized_image


def process_image(
    image,
    clusters,
    size,
    maxs,
    mins,
):
    quantized = quantize_colors(image, clusters)
    resized = resize_image(quantized, size)
    tiled = tile_image(resized, max_size=maxs, min_size=mins)
    similarity = ssim(
        resize_image(image, size),
        tiled,
        channel_axis=-1,
        gaussian_weights=True,
        multichannel=True,
    )
    return tiled, similarity


def show_image(img_url):
    """
    Shows processed image above unprocessed image for testing purposes.
    Args:
        img_url (string): path to image file.
    """
    img = cv2.imread(img_url)
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        img_rgb = img
    processed_image = process_image(
        img_rgb, CLUSTERS, SIZE, MAX_TILE_SIZE, MIN_TILE_SIZE
    )
    fig, ax = plt.subplots(2)
    ax[1].imshow(processed_image)
    ax[0].imshow(img_rgb)
    ax[0].axis("off")
    ax[1].axis("off")
    plt.show()
    print(
        f"Structural Similarity Index Measure: "
        f"{ssim(resize_image(img_rgb, SIZE), processed_image, channel_axis=-1,
                gaussian_weights=True, multichannel=True)}"
    )


app = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label="Input Image Component"),
        gr.Number(
            label="How many colors to auto select?", value=8, minimum=1, maximum=256
        ),
        gr.Number(label="Size of image's short edge", value=256, minimum=32),
        gr.Number(label="Max tile size"),
        gr.Number(label="Min tile size"),
    ],  # type= parameter not set. Defaults to numpy.array
    outputs=[
        gr.Image(label="Output Image Component"),
        gr.Textbox(label="Structural Similarity Index Mesaure"),
    ],
)

app.launch()

# if __name__ == "__main__":
#     start = datetime.now()
#     show_image("test_images/linus_and_card.jpg")
#     end = datetime.now()
#     print(f"elapsed time: {end - start}")
