import pygame
import sys
import random
from math import cos, sin, pi, sqrt, atan
import numpy as np
import time


COLORS = {
    'WHITE':    (255, 255, 255),
    'BLACK':    (0, 0, 0),
    "RED":      (255, 0, 0),
    "GREEN":    (0, 255, 0),
    "BLUE":     (0, 0, 255),
}


class Point:

    POINT_NUMBER = 0

    def __init__(self, r, angle, color):
        if r < 0:
            print('r should be always positive!')
            print(f'Rectifying, r = {r} ==> r = {abs(r)}, angle = {angle} ==> angle = {angle + pi}')
            r = abs(r)
            angle += pi

        self.number = Point.POINT_NUMBER
        self.len = r
        self.ang = angle
        self.color = color

        Point.POINT_NUMBER += 1

    def polar_to_cart(self):
        return self.len * cos(self.ang), self.len * sin(self.ang)

    def transform(self):
        x, y = self.polar_to_cart()
        transformed_x, transformed_y = x + WIDTH // 2, HEIGHT // 2 - y
        return transformed_x, transformed_y


ORIGIN = Point(0, 0, COLORS['WHITE'])


def main():
    print(sqrt((645 - 606.61) ** 2 + (390 - 396.85) ** 2))


def launch(w, h):
    print('Launching pygame. Welcome!')
    pygame.init()

    s = pygame.display.set_mode((w, h))
    pygame.display.set_caption('RobotArm')

    return s


def display_initials(s, W, H):
    s.fill('White')

    pygame.draw.line(s, COLORS['BLACK'], (W // 2, 0), (W // 2, H), width=5)
    pygame.draw.line(s, COLORS['BLACK'], (0, H // 2), (W, H // 2), width=5)
    pygame.display.update()

    pygame.display.update()


def initialize_points(N):
    while N > len(COLORS) - 2:

        N = int(input(f'N value needs to be less than {len(COLORS)}: '))

    pts = []
    NON_USED_COLORS = list(COLORS.keys())[2:]
    for _ in range(N):
        index = random.randint(0, len(NON_USED_COLORS) - 1)
        pts.append(Point(random.randint(100, 200), random.uniform(0, 2 * pi), COLORS[NON_USED_COLORS[index]]))
        NON_USED_COLORS.pop(index)
    # return [Point(247, 4.14)]
    return pts


def display_points(s, points, display=False):
    l = [ORIGIN.transform()]

    for point in points:
        x, y = point.polar_to_cart()
        x_1, y_1 = l[-1]
        l.append((x_1 + round(x, 2), y_1 - round(y, 2)))

        pygame.draw.line(s, point.color, l[-1], l[-2], width=3)

    if display:
        display_transformation(points)
    pygame.display.update()


def display_transformation(points):

    l = [ORIGIN.transform()]
    for point in points:
        x, y = point.polar_to_cart()
        x_1, y_1 = l[-1]
        x_i, y_i = x_1 + x, y_1 - y
        l.append((x_i, y_i))

        print(
            f'Transforming point {point.number}: r = {round(point.len, 2)}, theta = {round(point.ang, 2)} to x = {round(x_i, 2)}, y = {round(y_i, 2)}')


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


def gradientDescent(l_r, _steps, points, target, _s, _w, _h, epsilon=0.0001):
    def update_thetas(pts, thetas):
        for index, point in enumerate(pts):
            point.ang = thetas[index, 0]

    def extract_features(pts):
        lens, thetas = [], []
        for point in pts:
            lens.append(point.len)
            thetas.append(point.ang)

        return np.expand_dims(np.array(lens), 1), np.expand_dims(np.array(thetas), 1)

    def computeCost(features, t):
        r_features, theta_features = features
        r_target, theta_target = t.len, t.ang

        x = np.multiply(r_features, np.cos(theta_features))
        y = np.multiply(r_features, np.sin(theta_features))

        COS = np.sum(x) - r_target * cos(theta_target)
        SIN = np.sum(y) - r_target * sin(theta_target)

        return 1 / 2 * (COS ** 2 + SIN ** 2), COS * y - SIN * x

    for step in range(_steps):
        lengths, angles = extract_features(points)
        J, G = computeCost((lengths, angles), target)
        print(f'Step {step} | J = {round(J, 2)}')
        if J < epsilon:
            break
        angles = np.subtract(angles, - l_r * G)
        update_thetas(points, angles)
        display_initials(_s, _w, _h)
        display_points(_s, points)


def test(w, h):
    screen = launch(w, h)
    n = 2
    display_initials(screen, w, h)
    points = initialize_points(n)
    display_points(screen, points)
    clock = pygame.time.Clock()
    lr, steps = 0.00001, 4000
    while 1:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = pygame.mouse.get_pos()
                r, theta = cart_to_polar(x - w // 2, h // 2 - y)
                p = Point(r, theta, COLORS['BLACK'])
                gradientDescent(lr, steps, points, p, screen, w, h)

        pygame.display.update()
        clock.tick(60)


if __name__ == '__main__':
    WIDTH, HEIGHT = 1290, 780
    # main()
    test(WIDTH, HEIGHT)
