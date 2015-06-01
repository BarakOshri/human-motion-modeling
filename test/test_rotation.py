import sys
sys.path.append('/deep/u/kuanfang/action-prediction/');
from utils.rotation import *

vec = [1, 5, 3]
quat = r3_to_quat(vec)
print quat.q
vec_recons = quat_to_r3(quat)
print vec_recons
quat_recons = r3_to_quat(vec_recons)
print quat_recons.q
vec_recons2 = quat_to_r3(quat_recons)
print vec_recons2

print '***'

vec3 = vec_recons2 + 0.01
print vec3
quat3 = r3_to_quat(vec3)
print quat3.q
vec4 = quat_to_r3(quat3)
print vec4

