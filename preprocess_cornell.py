# Preprocess and then save the Cornell CAD-120 dataset
from util.cornell_utils import *
from util.space import *
from util.mocap_utils import *
import os
import numpy as np
from time import clock

tic = clock()

path_dataset = 'data/cornell/'

# Read Data
pos, pos_conf, ori, ori_conf, subject \
    = read(os.path.join(path_dataset, 'Subject1_annotations'))

# Extract the starting/ending frame number under specified condition
list_reaching = []
tot_len = 0

for activity_label, activities in subject.iteritems():
    for id in activities.keys():
        activity = activities[id]
        sub_activities = activity['sub_activities']
        for sub_activity in sub_activities:
            if sub_activity['sub_activity_id'] == 'reaching': # TODO
                list_reaching.append((sub_activity['start_frame'], 
                                        sub_activity['end_frame']))
                tot_len += \
                    (sub_activity['end_frame'] - sub_activity['start_frame'])

# Extract the Sequences
pos = np.concatenate([pos[start:end, :] for (start, end) in list_reaching],
                        axis=0)
ori = np.concatenate([ori[start:end, :] for (start, end) in list_reaching],
                        axis=0)
list_reaching_new = []
cnt = 0
for (start, end) in list_reaching:
    list_reaching_new.append([cnt, cnt+end-start])
    cnt += (end-start)

index_reaching = np.array(list_reaching_new)

# From Y-up to Z-up Coordinate System
pos, ori = change_space(pos, ori, R=np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

# Preprocess
print 'Preprocessing the joint of interests...'

data_joi, mean_joi, std_joi, pos_joi = \
    preprocess_joi(joint_idx, ['left_hand', 'right_hand'], pos, ori)
pos_joi_recon = postprocess_joi(joint_idx, data_joi, mean_joi, std_joi)

print 'done.'

print 'Preprocessing the relative positions...'

data_relpos, mean_relpos, std_relpos = \
    preprocess_relpos(joint_idx, connection, pos, ori)
pos_recon = \
    postprocess_relpos(joint_idx, connection, 
                        data_relpos, mean_relpos, std_relpos)

print 'done.'

# Save and print
np.save(os.path.join(path_dataset, 'index'), index_reaching)
np.save(os.path.join(path_dataset, 'pos'), pos)
np.save(os.path.join(path_dataset, 'ori'), ori)
np.save(os.path.join(path_dataset, 'data_joi'), data_joi)
np.save(os.path.join(path_dataset, 'mean_joi'), mean_joi)
np.save(os.path.join(path_dataset, 'std_joi'), std_joi)
np.save(os.path.join(path_dataset, 'pos_joi'), pos_joi)
np.save(os.path.join(path_dataset, 'data_relpos'), data_relpos)
np.save(os.path.join(path_dataset, 'mean_relpos'), mean_relpos)
np.save(os.path.join(path_dataset, 'std_relpos'), std_relpos)

toc = clock()
print 'Done in {} secs.'.format(toc-tic)

print 'Reoncstruction error of joints of interest: {}'\
        .format(np.mean((pos_joi - pos_joi_recon)**2))
print 'Reoncstruction error of relative position: {}'\
        .format(np.mean((pos - pos_recon)**2))

# for t in range(100):
#     pp = [np.linalg.norm(data_relpos[t, 3*i:3*(i+1)])\
#             for i in range(data_relpos.shape[1]/3)]
#     print pp

