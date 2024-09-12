import numpy as np


class Body:
    def __init__(self):
        self._is_static = None
        return

    def contains(self, pt) -> bool:
        # Implementation of membership operation.
        # return True if a given point belongs to the obstacle
        raise NotImplementedError

    def entry_time(self, pt, ray) -> float:
        # Implementation of the hitting time: given a ray p_t = x + t v, \quad t \geq 0, it returns
        # \inf\{t \geq 0 : p_t \in O \}.
        # In general, computation of the entry time reduces to a simple convex optimization problem when $O$ is convex.
        raise NotImplementedError

    def distance(self, pt):
        # Implementation of the distance function: given a point x, it returns
        # d(x, O) = \inf\{d(x, y) : y \in O\}.
        raise NotImplementedError


class CircularBody(Body):
    def __init__(self, center, radius):
        super(CircularBody, self).__init__()
        self._center = center
        self._radius = radius

    def contains(self, pt) -> bool:
        return True if ((pt - self._center) ** 2).sum() <= (1.3 * self._radius) ** 2 else False

    def entry_time(self, pt, ray) -> float:
        if self.contains(pt):
            # if the starting point is in the circle, the entry time is 0 by def
            return 0.
        a = (ray ** 2).sum()
        b = ((self._center - pt) * ray).sum()
        c = ((self._center - pt) ** 2).sum() - self._radius ** 2

        d = b ** 2 - a * c
        if d < 0:
            # if discriminant = 0, the ray does not meet the circle
            return np.inf
        else:
            # solve quadratic equation to find out the entry time
            # a t^2 - 2b t + c = 0
            # if all solutions are negative, the entry time is $\infty$ by def
            sol = [(b - d ** .5) / a, b + d ** .5 / a]
            return min([t for t in sol if t > 0], default=np.inf)

    def distance(self, pt):
        return max(0, ((pt - self._center) ** 2).sum() ** .5 - self._radius)

    @property
    def center(self):
        return np.copy(self._center)

    @property
    def radius(self):
        return self._radius


class RectangularBody(Body):
    def __init__(self, lb, ub):
        super(RectangularBody, self).__init__()
        self._lb = lb
        self._ub = ub

    def contains(self, pt) -> bool:
        return True if (np.all(pt <= self._ub) and np.all(self._lb <= pt)) else False

    def entry_time(self, pt, ray) -> float:
        # TODO : implement the method(not needed right now...)
        return Body.entry_time(self, pt, ray)

    def distance(self, pt):
        # Note that if $R = \prod_i R_i$, then we have
        # $d(x, R) = \left( \sum_i d(x_i, R_i)^2  \right)^{1/2}$.
        return (np.maximum(np.maximum(self._lb - pt, pt - self._ub), 0.) ** 2).sum() ** .5

    @property
    def lb(self):
        return np.copy(self._lb)

    @property
    def ub(self):
        return np.copy(self._ub)


class DynamicBody(Body):
    def __init__(self, *args):
        super(DynamicBody, self).__init__()
        self._is_static = False

    def sim(self, *args):
        # one-step simulation of dynamic obstacle
        # must be implemented if the obstacle is dynamic
        raise NotImplementedError

    def reset(self, *args):
        # reset states of the body
        raise NotImplementedError

    def contains(self, pt) -> bool:
        raise NotImplementedError

    def entry_time(self, pt, ray) -> float:
        raise NotImplementedError

    def distance(self, pt):
        raise NotImplementedError


class StaticBody(Body):
    def __init__(self, *args):
        super(StaticBody, self).__init__()
        self._is_static = True

    def contains(self, pt) -> bool:
        raise NotImplementedError

    def entry_time(self, pt, ray) -> float:
        raise NotImplementedError

    def distance(self, pt):
        raise NotImplementedError