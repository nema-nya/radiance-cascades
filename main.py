import numpy as np
from PIL import Image

RESOLUTION = 2**5
CASCADES_N = 5
ANGULAR_RESOLUTION = 32


def sample(texture, x, y):
    # 0l 1r
    h, w, _ = texture.shape
    # 1280 740
    # 720  360
    # 3
    x = x * (w - 1) + 1 / 2
    y = y * (h - 1) + 1 / 2
    # 2x2 grid of px
    xq = np.floor(x).astype(int)
    xr = x - xq
    xp = xq + 1

    yq = np.floor(y).astype(int)
    yr = y - yq
    yp = yq + 1

    # wrapping without repeating
    xq = max(0, xq)
    yq = max(0, yq)
    xp = max(0, xp)
    yp = max(0, yp)

    xp = min(w - 1, xp)
    yp = min(h - 1, yp)
    xq = min(w - 1, xq)
    yq = min(h - 1, yq)

    # we would do mod instead here if we needed repeating

    top_left = texture[yq, xq]
    top_right = texture[yq, xp]
    bot_left = texture[yp, xq]
    bot_right = texture[yp, xp]

    top = top_left * (1 - xr) + top_right * xr
    bot = bot_left * (1 - xr) + bot_right * xr

    return top * (1 - yr) + bot * yr


def sample_cascade(cascade, x, y, theta):
    # 0l 1r
    h, w, s, _ = cascade.shape
    # 1280 740
    # 720  360
    # 3
    x = x * (w - 1) + 1 / 2
    y = y * (h - 1) + 1 / 2
    # 2x2 grid of px
    xq = np.floor(x).astype(int)
    xr = x - xq
    xp = xq + 1

    yq = np.floor(y).astype(int)
    yr = y - yq
    yp = yq + 1

    aq = np.floor(theta).astype(int)
    ar = theta - aq
    ap = aq + 1

    # wrapping without repeating
    xq = max(0, xq)
    yq = max(0, yq)
    xp = max(0, xp)
    yp = max(0, yp)

    xp = min(w - 1, xp)
    yp = min(h - 1, yp)
    xq = min(w - 1, xq)
    yq = min(h - 1, yq)

    aq = aq % s
    ap = ap % s

    top_left_prev = cascade[yq, xq, aq]
    top_left_next = cascade[yq, xq, ap]

    top_right_prev = cascade[yq, xp, aq]
    top_right_next = cascade[yq, xp, ap]

    bot_left_prev = cascade[yp, xq, aq]
    bot_left_next = cascade[yp, xq, ap]

    bot_right_prev = cascade[yp, xp, aq]
    bot_right_next = cascade[yp, xp, ap]

    top_prev = top_left_prev * (1 - xr) + top_right_prev * xr
    top_next = top_left_next * (1 - xr) + top_right_next * xr

    bot_prev = bot_left_prev * (1 - xr) + bot_right_prev * xr
    bot_next = bot_left_next * (1 - xr) + bot_right_next * xr

    prev = top_prev * (1 - yr) + bot_prev * yr
    next = top_next * (1 - yr) + bot_next * yr

    return prev * (1 - ar) + next * ar


# how much light is there on x,y cord
# L(x,y) -> light at that cord
# instead of asking that, we have l(x,y,theta)
# how much light reaches the cord xy at angle theta (multipole expansion)
# we take this l(x,y,theta) function and give 2 more arguments
# how much light comes to x,y at angle theta between near and far
# 0-8, if we know 0-4, 4-8 sum of light is the answer


def rad(texture, x, y, theta, near, far):
    p = np.array([x, y])
    r = np.array([np.cos(theta), np.sin(theta)])
    result = np.array([0, 0, 0], dtype=float)
    for i in range(int((far - near) * RESOLUTION)):
        q = p + r * (near + i / RESOLUTION)
        result += sample(texture, q[0], q[1])
    return result


def rad0(texture, x, y, theta):
    p = np.array([x, y])
    r = np.array([np.cos(theta), np.sin(theta)])
    q = p + r * (1 / RESOLUTION)
    return sample(texture, q[0], q[1])


def L(texture):
    h, w, c = texture.shape
    out = np.zeros_like(texture)
    cascade0 = np.zeros((h, w, 4, c))
    for i in range(texture.shape[0]):
        for j in range(texture.shape[1]):
            for k in range(4):
                theta = k / 4 * 2 * np.pi
                p = np.array([(i + 1 / 2) / h, (j + 1 / 2) / w])
                r = np.array([np.cos(theta), np.sin(theta)])
                q = p + r * (1 / RESOLUTION)
                cascade0[i, j, k] = sample(texture, q[0], q[1])
    cascades = [cascade0]
    near = 1
    far = 2
    for _ in range(1, CASCADES_N):
        for theta in np.linspace(0, 2 * np.pi, ANGULAR_RESOLUTION)[:-1]:
            # result += rad(texture, x, y, theta, near / RESOLUTION, far / RESOLUTION)
            pass
        near = far
        far = 2 * near
    out = cascade0.mean(axis=2)
    return out


def main():
    im = Image.open("light.png").resize((64, 64)).convert("RGB")
    im = np.array(im)
    im = im.astype(float) / 255.0

    out = L(im)

    out /= out.max() + 1e-4
    out = np.clip((out * 255.0).astype(int), 0, 255).astype(np.uint8)
    out = Image.fromarray(out, "RGB")
    out.save("light_field.png")
    # v = np.arange(10)
    # print(interpolate(v, 0, 20))


if __name__ == "__main__":
    main()
