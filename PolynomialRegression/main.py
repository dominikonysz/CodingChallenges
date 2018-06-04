import tensorflow as tf
import pygame
from pygame import gfxdraw

import random as r

# pygame variabels
screen = 0
screen_size = 500, 400

# general variabels
points = []
curve = []

func_degree = 6

x = tf.placeholder('float')
y = tf.placeholder('float')

sess = tf.Session()


def init():
    global screen

    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption('Polynomial Regression Demo')

def main():
    global line_y, points, curve
    init()
    clock = pygame.time.Clock()

    with tf.Session() as sess:
        pred = model(x)
        cost = tf.reduce_mean(tf.square(pred - y))
        optimizer = tf.train.AdamOptimizer(0.05).minimize(cost)
        sess.run(tf.global_variables_initializer())

        running = True
        while running:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        running = False
                        break
                    if event.key == pygame.K_r:
                        points = []
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    points.append(pos)
                if pygame.mouse.get_pressed()[0] == True:
                    pos = pygame.mouse.get_pos()
                    points.append(pos)


            if len(points) > 0:
                xs = [(x/screen_size[0]) * 2 - 1 for x, _ in points]
                ys = [((screen_size[1] - y)/screen_size[1]) * 2 - 1 for _, y in points]
                _, c = sess.run([optimizer, cost], feed_dict={x: xs,
                                                              y: ys})
                print(c)
            curve = []
            for xs in frange(-1, 1.01, 0.04):
                curve.append(((xs + 1)/2 * screen_size[0], (1 - ((float(sess.run(pred, feed_dict={x: xs})) + 1) / 2)) * screen_size[1]))

            draw()
            pygame.display.update()

def model(x):
    # create model
    co = []
    for i in range(func_degree):
        co.append(tf.Variable([r.random() * 2 - 1]))
    # compute output
    out = 0
    for i in range(len(co)):
        out += co[i] * x**i
    return out


def draw():
    global screen, curve
    screen.fill((0, 0, 0))
    for point in points:
        gfxdraw.aacircle(screen, point[0], point[1], 4, (255, 255, 255))
        gfxdraw.filled_circle(screen, point[0], point[1], 4, (255, 255, 255))

    if len(curve) > 1:
        pygame.draw.aalines(screen, (255, 255, 255), False, curve, True)

def frange(start, stop, step):
    while start < stop:
        yield start
        start += step

if __name__=='__main__':
    main()
