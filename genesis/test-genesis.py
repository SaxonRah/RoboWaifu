import platform
import genesis as gs
gs.init(backend=gs.cpu)


if platform.system() == "Windows":
    print("Running on Windows")
    scene = gs.Scene(show_viewer=False)
elif platform.system() == "Linux":
    print("Running on Linux")
    scene = gs.Scene(show_viewer=True)
else:
    print("Running on another operating system")
    scene = gs.Scene(show_viewer=False)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

scene.build()

for i in range(1000):
    scene.step()
