import pygame
import sys
import random
from math import cos, sin, pi, sqrt, atan
import numpy as np


COLORS = {
    'WHITE':    (255, 255, 255),
    'BLACK':    (0, 0, 0),
    "RED":      (255, 0, 0),
    "GREEN":    (0, 255, 0),
    "BLUE":     (0, 0, 255),
}


class Point:

    POINT_NUMBER = 0

    def __init__(self, r, angle):
        if r < 0:
            print('r should be always positive!')
            print(f'Rectifying, r = {r} ==> r = {abs(r)}, angle = {angle} ==> angle = {angle + pi}')
            r = abs(r)
            angle += pi

        self.number = Point.POINT_NUMBER
        self.len = r
        self.ang = angle
        Point.POINT_NUMBER += 1

    def polar_to_cart(self):
        return self.len * cos(self.ang), self.len * sin(self.ang)

    def transform(self):
        x, y = self.polar_to_cart()
        transformed_x, transformed_y = x + WIDTH // 2, HEIGHT // 2 - y
        return transformed_x, transformed_y


ORIGIN = Point(0, 0)


def main():
    print(sqrt((645 - 606.61) ** 2 + (390 - 396.85) ** 2))


def launch(w, h):
    print('Launching pygame. Welcome!')
    pygame.init()

    s = pygame.display.set_mode((w, h))
    pygame.display.set_caption('RobotArm')

    return s


def display_initials(s, W, H, n):
    s.fill('White')

    pygame.draw.line(s, COLORS['BLACK'], (W // 2, 0), (W // 2, H), width=5)
    pygame.draw.line(s, COLORS['BLACK'], (0, H // 2), (W, H // 2), width=5)
    pygame.display.update()

    points = initialize_points(n)
    display_points(s, points)
    pygame.display.update()
    return points


def initialize_points(N):
    while N > len(COLORS) - 2:
        N = int(input(f'N value needs to be less than {len(COLORS)}: '))

    # return [Point(247, 4.14)]
    return [Point(random.randint(100, 300), random.uniform(0, 2 * pi)) for _ in range(N)]


def display_points(s, points):
    l = [ORIGIN.transform()]
    NON_USED_COLORS = list(COLORS.keys())[2:]
    for point in points:
        x, y = point.polar_to_cart()
        x_1, y_1 = l[-1]
        l.append((x_1 + round(x, 2), y_1 - round(y, 2)))
        index = random.randint(0, len(NON_USED_COLORS) - 1)
        pygame.draw.line(s, COLORS[NON_USED_COLORS[index]], l[-1], l[-2], width=3)
        NON_USED_COLORS.pop(index)

    print(l)
    display_transformation(points)
    pygame.display.update()


def display_transformation(points):
    for point in points:
        x_i, y_i = point.transform()
        print(f'Transforming point {point.number}: r = {round(point.len, 2)}, theta = {round(point.ang, 2)} to x = {round(x_i, 2)}, y = {round(y_i, 2)}')


def cart_to_polar(x, y):
    if x == 0:
        if y != 0:
            return y, pi / 2
        else:
            return 0, 0

    elif x > 0:
        return sqrt(x * x + y * y), atan(y / x)

    else:
        return sqrt(x * x + y * y), atan(y / x) + pi


def extract_features(points):
    lengths, angles = [], []
    for point in points:
        lengths.append(point.len)
        angles.append(point.ang)

    return np.expand_dims(np.array(lengths), 1), np.expand_dims(np.array(angles), 1)


def computeCost(features, target):
    r_features, theta_features = features
    r_target, theta_target = target.len, target.ang
    print(r_features)
    print(theta_features)
    print(r_target)
    print(theta_target)

    x = np.multiply(r_features, np.cos(theta_features))
    y = np.multiply(r_features, np.sin(theta_features))

    COS = np.sum(x) - r_target * cos(theta_target)
    SIN = np.sum(y) - r_target * sin(theta_target)
    J = 1 / 2 * (COS ** 2 + SIN ** 2)

    grad = SIN * x - COS * y

    return J, grad


def gradientDescent():
    pass


def test(w, h):
    screen = launch(w, h)
    n = 3
    points = display_initials(screen, w, h, n)
    clock = pygame.time.Clock()
    while 1:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = pygame.mouse.get_pos()
                r, theta = cart_to_polar(x - w // 2, h // 2 - y)
                p = Point(r, theta)
                display_points(screen, [p])
                print(computeCost(extract_features(points), p))

        pygame.display.update()
        clock.tick(60)


if __name__ == '__main__':
    WIDTH, HEIGHT = 1290, 780
    # main()
    test(WIDTH, HEIGHT)
