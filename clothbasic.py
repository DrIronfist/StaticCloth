import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

N = 10
kS = ti.field(dtype=ti.f32, shape=())
kS[None] = 50
springDist = 1
dt = 1.0/500.0
g = ti.Vector.field(3, ti.f32, shape=())
g[None] = tm.vec3([0, -0.1, 0])

kD = ti.field(dtype=ti.f32, shape=())
kD[None] = 0.1
particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N * N)
Point = ti.types.struct(pos=tm.vec3, vel=tm.vec3)

points = ti.Struct.field({
    "pos": tm.vec3,
    "prevPos": tm.vec3,
    "prevVel": tm.vec3
}, shape=(N, N))

# points = ti.Vector.field(6, dtype=ti.f32, shape=(N, N))
lines = ti.Vector.field(3, dtype=ti.f32, shape=(4 * N * (N - 1)))


@ti.kernel
def init_points_pos(points: ti.template()):
    for i in range(points.shape[0]):
        points[i] = [i for j in ti.static(range(3))]


@ti.kernel
def initClothPoints():
    print("resetting")
    for x in range(N):
        for y in range(N):
            points[x, y].pos = ti.Vector([x, y, 0])
            points[x, y].prevPos = ti.Vector([x, y, 0])
            points[x, y].prevVel = ti.Vector([0, 0, 0])
            i = x * 10 + y
            particles_pos[i] = [x, y, 0]

@ti.kernel
def drawClothPoints():
    for x in range(N):
        for y in range(N):
            i = x * 10 + y
            particles_pos[i] = points[x, y].pos



@ti.kernel
def drawConnections():
    # Initialize vertical lines
    for x in range(N):
        for y in range(N - 1):
            line_idx = (x * (N - 1) + y) * 2
            lines[line_idx] = points[x, y].pos
            lines[line_idx + 1] = points[x, y + 1].pos

    # Initialize horizontal lines
    for x in range(N - 1):
        for y in range(N):
            line_idx = 2 * (N * (N - 1 + x) + y)
            lines[line_idx] = points[x, y].pos
            lines[line_idx + 1] = points[x + 1, y].pos




@ti.kernel
def updatePoints():
    k = kS[None]
    for x in range(N):
        for y in range(N):
            if (x != 0 or y != N - 1) and (x != N - 1 or y != N - 1):
                curr = points[x, y].pos
                force = g[None]
                if 0 < x < N - 1:
                    lPDisp = tm.distance(points[x - 1, y].pos, curr)
                    if lPDisp > 0:
                        force += k * (lPDisp - springDist) * tm.normalize(points[x - 1, y].pos - curr)
                    rPDisp = tm.distance(points[x + 1, y].pos, curr)
                    if rPDisp > 0:
                        force += k * (rPDisp - springDist) * tm.normalize(points[x + 1, y].pos - curr)
                elif x == 0:
                    rPDisp = tm.distance(points[x + 1, y].pos, curr)
                    if rPDisp > 0:
                        force += k * (rPDisp - springDist) * tm.normalize(points[x + 1, y].pos - curr)
                else:
                    lPDisp = tm.distance(points[x - 1, y].pos, curr)
                    if lPDisp > 0:
                        force += k * (lPDisp - springDist) * tm.normalize(points[x - 1, y].pos - curr)

                if 0 < y < N - 1:
                    uPDisp = tm.distance(points[x, y + 1].pos, curr)
                    if uPDisp > 0:
                        force += k * (uPDisp - springDist) * tm.normalize(points[x, y + 1].pos - curr)
                    dPDisp = tm.distance(points[x, y - 1].pos, curr)
                    if dPDisp > 0:
                        force += k * (dPDisp - springDist) * tm.normalize(points[x, y - 1].pos - curr)
                elif y == 0:
                    uPDisp = tm.distance(points[x, y + 1].pos, curr)
                    if uPDisp > 0:
                        force += k * (uPDisp - springDist) * tm.normalize(points[x, y + 1].pos - curr)
                else:
                    dPDisp = tm.distance(points[x, y - 1].pos, curr)
                    if dPDisp > 0:
                        force += k * (dPDisp - springDist) * tm.normalize(points[x, y - 1].pos - curr)
                force -= kD[None] * points[x, y].prevVel * points[x, y].prevVel
                newPrev = points[x, y].pos
                points[x, y].pos += points[x, y].pos - points[x, y].prevPos + force * dt * dt
                points[x, y].prevVel += force * dt
                points[x, y].prevPos = newPrev







window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position((N - 1) / 2, (N - 1) / 2, N * 2)
camera.lookat((N - 1) / 2, (N - 1) / 2, 0)
camera.up(0, 1, 0)
initClothPoints()

while window.running:
    camera.track_user_inputs(window, movement_speed=0.5, hold_key=ti.ui.RMB, yaw_speed=10, pitch_speed=10)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'r': initClothPoints()
    if window.is_pressed(ti.ui.UP):
        g[None].y += 0.001
        print(g[None].y)
    if window.is_pressed(ti.ui.DOWN):
        g[None].y -= 0.001
        print(g[None].y)
    if window.is_pressed(ti.ui.LEFT):
        kS[None] -= 0.001
    if window.is_pressed(ti.ui.RIGHT):
        kS[None] += 0.001
    if window.is_pressed('z'):
        kD[None] -= 0.001
    if window.is_pressed('x'):
        kD[None] += 0.001


    updatePoints()
    drawClothPoints()
    drawConnections()
    scene.particles(particles_pos, color=(0.68, 0.26, 0.19), radius=0.1)
    scene.lines(lines, color=(0.28, 0.68, 0.99), width=5.0)
    # Draw 3d-lines in the scene
    canvas.scene(scene)

    window.show()
