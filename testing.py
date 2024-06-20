import taichi as ti
ti.init(arch=ti.gpu)

val = ti.field( dtype=ti.f32, shape=())
val[None] = 0

@ti.kernel
def loopTest():
    a = (1 << 31) - 1
    for i in range(a):
        if(i == a - 1):
            val[None] = i


loopTest()
print(val[None])