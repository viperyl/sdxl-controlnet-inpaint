from typing import List
from PIL import Image


def cat_image_horizental(images: List[Image.Image]) -> Image.Image:
    '''
    concat image  horizentally
    refered URL: https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    '''
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset,0))
        x_offset += image.size[0]
    return new_image


def cat_image_to_grid(images: List[Image.Image], X: int, Y: int) -> Image.Image:
    '''
    concat image to X by Y grid
    '''
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (max_width, max_height))
    x_offset = 0
    y_offset = 0
    for x in range(X):
        for y in range(Y):
            cur_img = images[x*X + y]
            new_image.paste(cur_img, (x_offset, y_offset))
            x_offset += cur_img.size[0]
        x_offset = 0
        y_offset += cur_img.size[1]

    return images[0]