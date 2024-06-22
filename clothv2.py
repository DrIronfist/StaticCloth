import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

N = 50
kS = ti.field(dtype=ti.f32, shape=())
kS[None] = 10000
dt = 1.0 / 200.0
g = ti.Vector.field(3, ti.f32, shape=())
g[None] = tm.vec3([0, -9.8, 0])
sqrt2 = tm.sqrt(2)

kD = ti.field(dtype=ti.f32, shape=())
kD[None] = 0.5
elapsedTime = ti.field(dtype=ti.f32, shape=())
elapsedTime[None] = 0

pos = ti.Vector.field(3, ti.f32, shape=(N, N))
prevPos = ti.Vector.field(3, ti.f32, shape=(N, N))
velocity = ti.Vector.field(3, ti.f32, shape=(N, N))
locked = ti.field(dtype=ti.i8, shape=(N, N))
forces = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
indArray = ti.Vector.field(4, dtype=ti.i32, shape=((N - 1) * (4 * (N - 1) + 1)) + N - 1)
lengths = ti.field(dtype=ti.f32, shape=((N - 1) * (4 * (N - 1) + 1)) + N - 1)
particles = ti.Vector.field(3, dtype=ti.f32, shape=N * N)
lines = ti.Vector.field(3, dtype=ti.f32, shape=(2 * ((N - 1) * (4 * (N - 1) + 1))) + 2 * N - 2)
mesh_indices = ti.field(dtype=ti.i32, shape=(2 * (N - 1) * (N - 1) * 3))
triangle_normals = ti.Vector.field(3, dtype=ti.f32, shape=(2 * (N - 1) * (N - 1)))
vertex_normals = ti.Vector.field(3, dtype=ti.f32, shape=(N * N))


@ti.kernel
def initPoints():
    for i in range(N * N):
        x = i // N
        y = i % N
        pos[x, y] = [x, 0, y]
        prevPos[x, y] = [x, 0, y]
        velocity[x, y] = [0, 0, 0]
        locked[x, y] = ti.i8(0)
    locked[0, N - 1] = ti.i8(1)
    locked[N - 1, N - 1] = ti.i8(1)

    for x in range(N - 1):
        for y in range(N - 1):
            idx = x * (4 * (N - 1) + 1) + y * 4
            indArray[idx] = tm.ivec4(x, y, x + 1, y)
            lengths[idx] = ti.f32(1.0)
            indArray[idx + 1] = tm.ivec4(x, y, x, y + 1)
            lengths[idx + 1] = ti.f32(1.0)
            indArray[idx + 2] = tm.ivec4(x, y, x + 1, y + 1)
            lengths[idx + 2] = sqrt2
            indArray[idx + 3] = tm.ivec4(x, y + 1, x + 1, y)
            lengths[idx + 3] = sqrt2
        indArray[(x + 1) * (4 * (N - 1) + 1) - 1] = tm.ivec4(x, N - 1, x + 1, N - 1)
        lengths[(x + 1) * (4 * (N - 1) + 1) - 1] = ti.f32(1.0)
    for i in range(N - 1):
        idx = (N - 1) * (4 * (N - 1) + 1) + i
        indArray[idx] = tm.ivec4(N - 1, i, N - 1, i + 1)
        lengths[idx] = ti.f32(1.0)


    # mesh indices
    for x in range(N - 1):
        for y in range(N - 1):
            idx = 2 * (x * (N - 1) + y) * 3
            mesh_indices[idx] = x * N + y
            mesh_indices[idx + 1] = x * N + y + 1
            mesh_indices[idx + 2] = (x + 1) * N + y
            mesh_indices[idx + 3] = x * N + y + 1
            mesh_indices[idx + 4] = (x + 1) * N + y
            mesh_indices[idx + 5] = (x + 1) * N + y + 1


initPoints()
ti.sync()


@ti.kernel
def computeTriangleNormals():
    for x in range(N - 1):
        for y in range(N - 1):
            idx = 2 * (x * (N - 1) + y)
            a = pos[x, y]
            b = pos[x, y + 1]
            c = pos[x + 1, y]
            d = pos[x + 1, y + 1]
            # Triangle 1: a, b, c
            triangle_normals[idx] = tm.normalize(tm.cross(b - a, c - a))
            # Triangle 2: b, d, c
            triangle_normals[idx + 1] = tm.normalize(tm.cross(d - b, c - b))


@ti.kernel
def computeVertexNormals():
    # Reset vertex normals
    for x in range(N):
        for y in range(N):
            vertex_normals[x * N + y] = tm.vec3(0.0, 0.0, 0.0)

    for x in range(N - 1):
        for y in range(N - 1):
            idx = 2 * (x * (N - 1) + y)
            # Add the normal of triangle 1 to its vertices
            vertex_normals[x * N + y] += triangle_normals[idx]
            vertex_normals[x * N + y + 1] += triangle_normals[idx]
            vertex_normals[(x + 1) * N + y] += triangle_normals[idx]
            # Add the normal of triangle 2 to its vertices
            vertex_normals[x * N + y + 1] += triangle_normals[idx + 1]
            vertex_normals[(x + 1) * N + y] += triangle_normals[idx + 1]
            vertex_normals[(x + 1) * N + y + 1] += triangle_normals[idx + 1]

    # Normalize vertex normals
    for i in range(N * N):
        vertex_normals[i] = tm.normalize(vertex_normals[i])

@ti.kernel
def renderUpdate():
    for i in range(N * N):
        x = i // N
        y = i % N
        particles[i] = pos[x, y]

    for i in range(indArray.shape[0]):
        ind = 2 * i
        s = indArray[i]
        lines[ind] = pos[s.x, s.y]
        lines[ind + 1] = pos[s.z, s.w]



@ti.kernel
def update():
    grav = g[None]
    for i in ti.grouped(forces):
        forces[i] = grav
    ti.sync()
    k = kS[None]
    for i in range(indArray.shape[0]):
        s = indArray[i]
        pA = pos[s.x, s.y]
        pB = pos[s.z, s.w]
        length = lengths[i]
        lockA = locked[s.x, s.y]
        lockB = locked[s.z, s.w]
        l = tm.distance(pA, pB)
        if lockA == 0 and l != length:
            fDir = tm.normalize(pB - pA)
            forces[s.x, s.y] += k * (l - length) * fDir
        if lockB == 0 and l != length:
            fDir = tm.normalize(pA - pB)
            forces[s.z, s.w] += k * (l - length) * fDir
    ti.sync()
    t = elapsedTime[None]
    for i in range(N * N):
        x = i // N
        y = i % N
        position = pos[x, y]
        wind = tm.vec3(
            tm.sin(position.x * position.y * t),
            tm.cos(position.z * t),
            5 * tm.sin(position.x * t / N))
        lock = locked[x, y] == 1
        vel = velocity[x, y]
        prev = prevPos[x, y]
        if not lock:
            #forces[x, y] += wind
            if tm.length(vel) != 0:
                forces[x, y] -= kD[None] * vel
            newPrev = position
            position += position - prev
            position += forces[x, y] * dt * dt
            vel = (position - prev) / (2 * dt)
            prevPos[x, y] = newPrev
            pos[x, y] = position
            velocity[x, y] = vel


window = ti.ui.Window("Cloth", (768, 768))
canvas = window.get_canvas()
canvas.set_background_color((0.0, 0.0, 0.0))
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position((N - 1) / 2, (N - 1) / 2, N * 2)
camera.lookat((N - 1) / 2, (N - 1) / 2, 0)
camera.up(0, 1, 0)
sphere = ti.Vector.field(3, dtype=ti.f32, shape=(1))
sphere[0] = [0, 0, 0]

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
    elapsedTime[None] += dt

    computeTriangleNormals()

    computeVertexNormals()
    scene.particles(particles, color=(0.68, 0.26, 0.19), radius=0.1)
    # scene.particles(sphere, color=(0.1, 0.1, 0.1), radius=N/2)
    scene.lines(lines, color=(0.5, 0.5, 0.5), width=1.0)
    #scene.mesh_instance(vertices=particles, indices=mesh_indices, normals=vertex_normals, color=(0.68, 0.26, 0.19), two_sided=True, show_wireframe=False)
    canvas.scene(scene)
    window.show()
