import numpy as np
from PIL import Image

SCALE = 2.2

def light_field_slow(im, out):
    lights = set()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            l = im[i,j]
            if l[3] > 1e-2:
                lights.add((i,j))
            
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            u = np.array((i,j))
            color = np.zeros(4)
            for x,y in lights:
                v = np.array((x,y))
                l = im[x,y]
                d = np.linalg.norm(v-u) * SCALE
                if d < 1e-2 * SCALE:
                    continue
                color += l / d
            out[i,j] = color

    out /= out.max()
    return out 

def main():
    im = Image.open("light.png").resize((100, 100))
    im = np.array(im)
    im = im.astype(float) / 255.0
    out = np.zeros_like(im)

    out = light_field_slow(im, out)
    out = np.clip((out * 255.0).astype(int), 0, 255).astype(np.uint8)
    out = Image.fromarray(out, "RGBA")
    out.save("light_field.png")


if __name__ == "__main__":
    main()
