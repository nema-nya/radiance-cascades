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


def L(texture, x, y):
    result = np.array([0, 0, 0], dtype=float)
    for theta in np.linspace(0, 2 * np.pi, ANGULAR_RESOLUTION)[:-1]:
        near = 0
        far = 1
        for _ in range(CASCADES_N):
            result += rad(texture, x, y, theta, near / RESOLUTION, far / RESOLUTION)
            near = far
            far = 2 * near
    return result


def main():
    im = Image.open("light.png").resize((64, 64)).convert("RGB")
    im = np.array(im)
    im = im.astype(float) / 255.0
    out = np.zeros_like(im)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            out[i, j] = L(im, (j + 1 / 2) / im.shape[1], (i + 1 / 2) / im.shape[0])

    print(out.max())

    out /= out.max()
    out = np.clip((out * 255.0).astype(int), 0, 255).astype(np.uint8)
    out = Image.fromarray(out, "RGB")
    out.save("light_field.png")
    # v = np.arange(10)
    # print(interpolate(v, 0, 20))


if __name__ == "__main__":
    main()
