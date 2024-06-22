

import taichi as ti

ti.init(arch=ti.cpu)



@ti.kernel
def test():
    for i in range(30):
        print(i)

    for i in range(30, 60):
        print(i)


test()
