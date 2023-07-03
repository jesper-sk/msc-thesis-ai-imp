from math import pi


def deg_to_rad(deg: float):
    return (2 * pi * deg) / 360


def rad_to_deg(rad: float):
    return (rad * 360) / (2 * pi)
