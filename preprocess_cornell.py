# Preprocess and then save the Cornell CAD-120 dataset
from utils.cornell_utils import *
import os
import numpy as np

path_dataset = 'data/cornell/'

# read
pos, pos_conf, ori, ori_conf, subject \
    = read(os.path.join(path_dataset, 'Subject1_annotations'))

# extract the starting/ending frame number under specified condition
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

# extract the sequences
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

# preprocess pos and ori into desired data representation
torso = joint_idx['torso']
lhand = joint_idx['left_hand']
rhand = joint_idx['right_hand']
data = np.concatenate([pos[:, 3*torso:3*(torso+1)], 
                        pos[:, 3*rhand:3*(lhand+1)],
                        pos[:, 3*rhand:3*(rhand+1)]], axis=1)

# TODO

# save and print
np.save(os.path.join(path_dataset, 'index'), index_reaching)
np.save(os.path.join(path_dataset, 'pos'), pos)
np.save(os.path.join(path_dataset, 'ori'), ori)
np.save(os.path.join(path_dataset, 'data'), data)

print index_reaching
print pos.shape
print ori.shape
print data.shape

