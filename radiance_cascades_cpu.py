import numpy as np
from PIL import Image

CASCADES_N = 5


def sample(texture, x, y):
    # 0l 1r
    h, w, _ = texture.shape
    # 1280 740
    # 720  360
    # 3
    x = x * w - 1 / 2
    y = y * h - 1 / 2
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
    x = x * w - 1 / 2
    y = y * h - 1 / 2
    theta = theta * s - 1 / 2
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


def L(texture, selector=None):
    h, w, c = texture.shape
    length = 1 / h
    cascade0 = np.zeros((h, w, 4, c))
    for i in range(cascade0.shape[0]):
        for j in range(cascade0.shape[1]):
            for k in range(4):
                y = (i + 1 / 2) / cascade0.shape[0]
                x = (j + 1 / 2) / cascade0.shape[1]
                theta = (k + 1 / 2) / cascade0.shape[2]
                p = np.array([x, y])
                r = np.array([np.cos(theta * 2 * np.pi), np.sin(theta * 2 * np.pi)])
                q = p + r * length
                cascade0[i, j, k] = sample(texture, q[0], q[1])
    cascades = [cascade0]
    cascade_number = np.floor(np.log2(h)).astype(int)
    for _ in range(1, cascade_number):
        prev_cascade = cascades[-1]
        # number of rays grows only by 2, (even tho it can grow by number of 4 or more), but then the work is the
        # same as cascades0, or even if we have a number higher than 4 it would take more thn cascade0
        # the entire process takes up exactly twice as much memory as cascade0
        cascade = np.zeros(
            (
                prev_cascade.shape[0] // 2,
                prev_cascade.shape[1] // 2,
                prev_cascade.shape[2] * 2,
                c,
            )
        )
        for i in range(cascade.shape[0]):
            for j in range(cascade.shape[1]):
                for k in range(cascade.shape[2]):
                    y = (i + 1 / 2) / cascade.shape[0]
                    x = (j + 1 / 2) / cascade.shape[1]
                    theta = (k + 1 / 2) / cascade.shape[2]
                    # we want to compute 4-8, but len is 4, in prev step we computed cascades of length 2
                    # so from the perspective from point p we only know 2-4 which is useless
                    # so instead we need to ask 4-6 and 6-8, but those cascades need to be cascades 2-4
                    # from the perspetive of another pixel

                    p = np.array([x, y])
                    r = np.array([np.cos(theta * 2 * np.pi), np.sin(theta * 2 * np.pi)])

                    # case 16-32
                    # len = 8 (prev cascade)
                    # prev cascade 8-16
                    # we need to ask someone who is 8 units away about his 8-16 cascade
                    # and that will be our cascade from 16-24
                    # and we also need to ask someone who is 16 units away about his 8-16 cascade
                    # which will be our 24-32 cascade
                    # and we need to merge that
                    # and thats our 16-32 cascade
                    pa = p + r * length
                    pb = p + r * 2 * length
                    cascade[i, j, k] = sample_cascade(
                        prev_cascade,
                        pa[0],
                        pa[1],
                        theta,
                    ) + sample_cascade(
                        prev_cascade,
                        pb[0],
                        pb[1],
                        theta,
                    )

        cascades.append(cascade)
        length *= 2

    if selector is not None and 0 not in selector:
        cascades[0] = np.zeros_like(cascades[0])
    while len(cascades) > 1:
        ix = len(cascades) - 1
        far = cascades.pop()
        if selector is not None and ix not in selector:
            continue
        close = cascades.pop()

        for i in range(close.shape[0]):
            for j in range(close.shape[1]):
                for k in range(close.shape[2]):
                    y = (i + 1 / 2) / close.shape[0]
                    x = (j + 1 / 2) / close.shape[1]
                    theta = (k + 1 / 2) / close.shape[2]
                    close[i, j, k] = close[i, j, k] + sample_cascade(far, x, y, theta)

        cascades.append(close)

    return cascades[0].mean(axis=2)


def main():
    im = Image.open("light.png").resize((256, 256)).convert("RGB")
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
