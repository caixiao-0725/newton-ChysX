import warp as wp
wp.init()
from newton._src.geometry.collision_primitive import collide_box_box

_vec8f = wp.types.vector(8, wp.float32)
_mat83f = wp.types.matrix((8, 3), wp.float32)

@wp.kernel
def test_kernel(result_dist: wp.array(dtype=_vec8f), result_normal: wp.array[wp.vec3], result_count: wp.array[wp.int32]):
    box1_pos = wp.vec3(0.0, 0.0, 0.1)
    box2_pos = wp.vec3(0.0, 0.0, 0.339)
    box1_rot = wp.mat33(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0)
    box2_rot = wp.mat33(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0)
    box1_size = wp.vec3(0.1, 0.1, 0.1)
    box2_size = wp.vec3(0.1, 0.1, 0.1)
    dist, pos, normals = collide_box_box(box1_pos, box1_rot, box1_size, box2_pos, box2_rot, box2_size, 0.04)
    result_dist[0] = dist
    count = 0
    for i in range(8):
        if dist[i] < 1e30:
            count += 1
            result_normal[i] = normals[i]
    result_count[0] = count

rd = wp.zeros(1, dtype=_vec8f)
rn = wp.zeros(8, dtype=wp.vec3)
rc = wp.zeros(1, dtype=wp.int32)
wp.launch(test_kernel, dim=1, inputs=[rd, rn, rc])
wp.synchronize()
dists = rd.numpy()[0]
count = rc.numpy()[0]
print(f'Contact count: {count}')
for i in range(count):
    n = rn.numpy()[i]
    print(f'  c{i}: dist={dists[i]:.5f} normal=({n[0]:.3f},{n[1]:.3f},{n[2]:.3f})')
