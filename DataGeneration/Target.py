class Target:

    def __init__(self, inital_x: float):
        self._x = inital_x

    @property
    def x_vec(self) -> float:
        return self._x
