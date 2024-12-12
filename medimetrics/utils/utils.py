import numpy as np


def reflect_pad_with_mask(image: np.ndarray, mask: np.ndarray, pad_width: int = 1) -> np.ndarray:
    if mask is None:
        mask = np.ones_like(image)
    padded_mask = np.pad(mask.copy(), pad_width=pad_width, constant_values=0)
    padded_image = np.pad(image.copy(), pad_width=pad_width, constant_values=0)
    padded_image[padded_mask == 0] = 0

    # while there are still 0 values on the mask
    # continue filling the image
    while (padded_mask <= 0).sum() > 0:
        for x in range(padded_image.shape[0]):
            for y in range(padded_image.shape[1]):
                # find a masked pixel
                if padded_mask[x, y] <= 0:
                    # top -> down
                    offset = 1
                    # check if two pixels above are not masked
                    if (y - 2 >= 0) and (padded_mask[x, y - 1] > 0) and (padded_mask[x, y - 2] > 0):
                        # check if further pixels above are not masked and pixels below are masked
                        while (
                            (y - offset - 2 >= 0)
                            and (y + offset < padded_image.shape[1])
                            and (padded_mask[x, y - offset - 2] > 0)
                            and (padded_mask[x, y + offset] <= 0)
                        ):
                            offset += 1

                        # fill with all available pixels on reflected axis:
                        for fill_offset in range(offset):
                            padded_image[x, y + fill_offset] = padded_image[x, y - fill_offset - 2]
                            padded_mask[x, y + fill_offset] = 1

                    # bottom -> up
                    offset = 1
                    # check if two pixels below are not masked
                    if (y + 2 < padded_image.shape[1]) and (padded_mask[x, y + 1] > 0) and (padded_mask[x, y + 2] > 0):
                        # check if further pixels below are not masked and pixels above are masked
                        while (
                            (y + offset + 2 < padded_image.shape[1])
                            and (y - offset >= 0)
                            and (padded_mask[x, y + offset + 2] > 0)
                            and (padded_mask[x, y - offset] <= 0)
                        ):
                            offset += 1

                        # fill with all available pixels on reflected axis:
                        for fill_offset in range(offset):
                            padded_image[x, y - fill_offset] = padded_image[x, y + fill_offset + 2]
                            padded_mask[x, y - fill_offset] = 1

                    # right -> left
                    offset = 1
                    # check if two pixels right are not masked
                    if (x + 2 < padded_image.shape[0]) and (padded_mask[x + 1, y] > 0) and (padded_mask[x + 2, y] > 0):
                        # check if further pixels below are not masked and pixels above are masked
                        while (
                            (x + offset + 2 < padded_image.shape[0])
                            and (x - offset >= 0)
                            and (padded_mask[x + offset + 2, y] > 0)
                            and (padded_mask[x - offset, y] <= 0)
                        ):
                            offset += 1

                        # fill with all available pixels on reflected axis:
                        for fill_offset in range(offset):
                            padded_image[x - fill_offset, y] = padded_image[x + fill_offset + 2, y]
                            padded_mask[x - fill_offset, y] = 1

                    # left -> right
                    offset = 1
                    # check if two pixels above are not masked
                    if (x - 2 >= 0) and (padded_mask[x - 1, y] > 0) and (padded_mask[x - 2, y] > 0):
                        # check if further pixels above are not masked and pixels below are masked
                        while (
                            (x - offset - 2 >= 0)
                            and (x + offset < padded_image.shape[0])
                            and (padded_mask[x - offset - 2, y] > 0)
                            and (padded_mask[x + offset, y] <= 0)
                        ):
                            offset += 1

                        # fill with all available pixels on reflected axis:
                        for fill_offset in range(offset):
                            padded_image[x + fill_offset, y] = padded_image[x - fill_offset - 2, y]
                            padded_mask[x + fill_offset, y] = 1

    return padded_image


if __name__ == "__main__":
    # short test:

    a = np.arange(4).reshape(2, 2)
    print(a)

    a_padded = reflect_pad_with_mask(a, mask=None, pad_width=2)
    print(a_padded)
