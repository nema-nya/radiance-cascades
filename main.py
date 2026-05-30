import numpy as np
from PIL import Image

SCALE = 2.2


def light_field_slow(im, out):
    lights = set()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            l = im[i, j]
            if l[3] > 1e-2:
                lights.add((i, j))

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            u = np.array((i, j))
            color = np.zeros(4)
            for x, y in lights:
                v = np.array((x, y))
                l = im[x, y]
                d = np.linalg.norm(v - u) * SCALE
                if d < 1e-2 * SCALE:
                    continue
                color += l / d
            out[i, j] = color

    out /= out.max()
    return out


def trilinear_interpolation(input_buffer, output_buffer):
    neighbors = [(x, y, z) for x in range(2) for y in range(2) for z in range(2)]
    for i in range(output_buffer.shape[0]):
        for j in range(output_buffer.shape[1]):
            for k in range(output_buffer.shape[2]):
                i_ = i / output_buffer.shape[0] * input_buffer.shape[0]
                j_ = j / output_buffer.shape[1] * input_buffer.shape[1]
                k_ = k / output_buffer.shape[2] * input_buffer.shape[2]
                fx = i_ - int(i_)
                fy = j_ - int(j_)
                fz = k_ - int(k_)
                i_ = int(i_)
                j_ = int(j_)
                k_ = int(k_)
                for x, y, z in neighbors:
                    if i_ + x < 0 or i_ + x >= input_buffer.shape[0]:
                        continue
                    if j_ + y < 0 or j_ + y >= input_buffer.shape[1]:
                        continue
                    if k_ + z < 0 or k_ + z >= input_buffer.shape[2]:
                        continue
                    f = (
                        1
                        * (1 - fx if x == 0 else fx)
                        * (1 - fy if y == 0 else fy)
                        * (1 - fz if z == 0 else fz)
                    )
                    output_buffer[i, j, k] += input_buffer[i_, j_, k_] * f


def light_field_fast(im, out):
    buffer = np.zeros((im.shape[0], im.shape[1], 4, 4))
    buffer[:-1, :, 2] = im[1:, :]  # top
    buffer[1:, :, 0] = im[:-1, :]  # bottom
    buffer[:, :-1, 1] = im[:, 1:]  # right
    buffer[:, 1:, 3] = im[:, :-1]  # left
    cascades = [buffer]
    n = 0
    while min(cascades[-1].shape) // 2 > 0:
        n += 1
        next_cascade = np.zeros(
            (
                cascades[-1].shape[0] // 2,
                cascades[-1].shape[1] // 2,
                2 * cascades[-1].shape[2],
                4,
            )
        )
        for i in range(next_cascade.shape[0]):
            for j in range(next_cascade.shape[1]):
                for k in range(next_cascade.shape[2]):
                    angle = k / next_cascade.shape[2] * 2 * np.pi
                    center = np.array(
                        [
                            (i + 1 / 2) / next_cascade.shape[0] * im.shape[0],
                            (j + 1 / 2) / next_cascade.shape[1] * im.shape[1],
                        ]
                    )
                    ray = np.array([np.cos(angle), np.sin(angle)])
                    near = 2 ** (n - 1)
                    far = 2**n
                    color = np.zeros(4)
                    for s in range(near, far):
                        target = center + s * ray
                        sample_dist = np.linalg.norm(target - center)
                        target[0] /= im.shape[0]
                        target[1] /= im.shape[1]
                        target[0] *= cascades[-1].shape[0]
                        target[1] *= cascades[-1].shape[1]
                        top = int(target[0])
                        left = int(target[1])
                        prev_rotated_angle = k // 2
                        neighbors = [
                            (x, y, z)
                            for x in range(2)
                            for y in range(2)
                            for z in range(2)
                        ]
                        for x, y, z in neighbors:
                            if top + x < 0 or top + x >= cascades[-1].shape[0]:
                                continue
                            if left + y < 0 or left + y >= cascades[-1].shape[1]:
                                continue

                            color_fx = target[0] - top
                            color_fy = target[1] - left
                            color_fz = k - 2 * prev_rotated_angle
                            color_f = (
                                1
                                * (1 - color_fx if x == 0 else color_fx)
                                * (1 - color_fy if y == 0 else color_fy)
                                * (1 - color_fz if z == 0 else color_fz)
                            )
                            color += (
                                cascades[-1][
                                    top + x,
                                    left + y,
                                    (prev_rotated_angle + z) % cascades[-1].shape[2],
                                ]
                                / sample_dist
                                * color_f
                            )
                    next_cascade[i, j, k] = color
        cascades.append(next_cascade)

    for n in range(1, len(cascades)):
        rn = len(cascades) - 1 - n
        tmp_buffer = np.zeros_like(cascades[rn - 1])
        trilinear_interpolation(cascades[rn], tmp_buffer)
        tmp_buffer += cascades[rn - 1]
        cascades[rn - 1] = tmp_buffer

    buffer = cascades[0]
    out = buffer.sum(2)
    return out


def main():
    im = Image.open("light.png").resize((100, 100))
    im = np.array(im)
    im = im.astype(float) / 255.0
    out = np.zeros_like(im)

    out = light_field_fast(im, out)
    out = np.clip((out * 255.0).astype(int), 0, 255).astype(np.uint8)
    out = Image.fromarray(out, "RGBA")
    out.save("light_field.png")


if __name__ == "__main__":
    main()
