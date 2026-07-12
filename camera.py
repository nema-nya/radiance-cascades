import numpy as np

# https://github.com/g-truc/glm


def get_perspective_mat(fov, aspect, near, far):
    f = 1.0 / np.tan(fov / 2.0)
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far / (far - near), -far * near / (far - near)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def get_look_at_mat(camera, target, up):
    z_ = target - camera
    z_ = z_ / np.linalg.norm(z_)

    x_ = np.cross(up, z_)
    x_ = x_ / np.linalg.norm(x_)

    y_ = np.cross(z_, x_)
    return np.array(
        [
            [x_[0], x_[1], x_[2], -np.dot(x_, camera)],
            [y_[0], y_[1], y_[2], -np.dot(y_, camera)],
            [z_[0], z_[1], z_[2], -np.dot(z_, camera)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class Camera:
    def __init__(self, position, target, aspect, up=None, fov=80.0, near=1.0, far=10.0):
        self.position = position
        self.target = target
        self.aspect = aspect
        if up is None:
            up = np.array([0.0, 1.0, 0.0])
        self.up = up
        self.fov = fov
        self.near = near
        self.far = far

    def view_mat(self):
        return get_look_at_mat(self.position, self.target, self.up)

    def projection_mat(self):
        return get_perspective_mat(self.fov, self.aspect, self.near, self.far)
