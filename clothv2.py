import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.cpu)

vec3 = tm.vec3
N = 10
kS = ti.field(dtype=ti.f32, shape=())
kS[None] = 1000
dt = 1.0 / 500.0
g = ti.Vector.field(3, ti.f32, shape=())
g[None] = tm.vec3([0, -9.8, 0])

kD = ti.field(dtype=ti.f32, shape=())
kD[None] = 1


@ti.dataclass
class Point:
    position: vec3
    prevPosition: vec3
    locked: bool
    vel: vec3


@ti.dataclass
class Stick:
    ind_a: tm.ivec2
    ind_b: tm.ivec2
    length: ti.f32


points = Point.field(shape=(N, N))
testPoints = ti.Vector.field(3, ti.f32, shape=(N, N))
sticks = Stick.field(shape=(2 * N * (N - 1)))
particles = ti.Vector.field(3, dtype=ti.f32, shape=N * N)
lines = ti.Vector.field(3, dtype=ti.f32, shape=(4 * N * (N - 1)))
forces = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))


@ti.kernel
def initPoints():

    for x, y in points:
        points[x, y].position = [x, y, 0]
        points[x, y].prevPosition = [x, y, 0]
        points[x, y].vel = [0, 0, 0]
        points[x, y].locked = False
    points[0, N - 1].locked = True
    points[N - 1, N - 1].locked = True

    # vertical sticks
    for x in range(N):
        for y in range(N - 1):
            lines_idx = x * (N - 1) + y
            sticks[lines_idx].ind_a = tm.ivec2(x, y)
            sticks[lines_idx].ind_b = tm.ivec2(x, y + 1)
            sticks[lines_idx].length = ti.f32(1.0)

    # vertical sticks
    for x in range(N - 1):
        for y in range(N):
            lines_idx = N * (N - 1 + x) + y
            sticks[lines_idx].ind_a = tm.ivec2(x, y)
            sticks[lines_idx].ind_b = tm.ivec2(x + 1, y)
            sticks[lines_idx].length = ti.f32(1.0)


initPoints()
ti.sync()


@ti.kernel
def renderUpdate():
    for x in range(N):
        for y in range(N):
            i = x * 10 + y
            particles[i] = points[x, y].position

    for i in range(2 * N * (N - 1)):
        ind = 2 * i
        s = sticks[i]
        pA = points[s.ind_a.x, s.ind_a.y]
        pB = points[s.ind_b.x, s.ind_b.y]
        lines[ind] = pA.position
        lines[ind + 1] = pB.position


@ti.kernel
def update():
    for x in range(N):
        for y in range(N):
            forces[x, y] = g[None]
    ti.sync()
    k = kS[None]
    for i in range(sticks.shape[0]):
        s = sticks[i]
        pA = points[s.ind_a.x, s.ind_a.y]
        pB = points[s.ind_b.x, s.ind_b.y]
        l = tm.distance(pA.position, pB.position)
        if not pA.locked and l != s.length:
            fDir = tm.normalize(pB.position - pA.position)
            forces[s.ind_a.x, s.ind_a.y] += k * (l - s.length) * fDir
        if not pB.locked and l != s.length:
            fDir = tm.normalize(pA.position - pB.position)
            forces[s.ind_b.x, s.ind_b.y] += k * (l - s.length) * fDir
    ti.sync()
    for i in range(N * N):
        x = i // N
        y = i % N
        p = points[x, y]
        if not p.locked:
            if tm.length(p.vel) != 0:
                forces[x, y] -= kD[None] * p.vel
            newPrev = p.position
            p.position += p.position - p.prevPosition
            p.position += forces[x, y] * dt * dt
            p.vel = (p.position - p.prevPosition)/(2 * dt)
            p.prevPosition = newPrev
            points[x, y] = p


window = ti.ui.Window("Cloth", (768, 768))
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position((N - 1) / 2, (N - 1) / 2, N * 2)
camera.lookat((N - 1) / 2, (N - 1) / 2, 0)
camera.up(0, 1, 0)

while window.running:
    camera.track_user_inputs(window, movement_speed=0.5, hold_key=ti.ui.RMB, yaw_speed=10, pitch_speed=10)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'r': initPoints()
    if window.is_pressed(ti.ui.UP):
        kS[None] += 0.001
    if window.is_pressed(ti.ui.DOWN):
        kS[None] -= 0.001

    update()
    renderUpdate()
    scene.particles(particles, color=(0.68, 0.26, 0.19), radius=0.1)
    scene.lines(lines, color=(0.28, 0.68, 0.99), width=1.0)
    canvas.scene(scene)
    window.show()
