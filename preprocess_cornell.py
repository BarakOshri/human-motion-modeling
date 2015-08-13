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
pos_arr, posconf_arr, ori_arr, oriconf_arr, subject \
    = read(os.path.join(path_dataset, 'Subject1_annotations'))

# Extract the starting/ending frame number under specified condition
list = []
tot_len = 0

for activity_label, activities in subject.iteritems():
    for id in activities.keys():
        activity = activities[id]
        sub_activities = activity['sub_activities']
        for sub_activity in sub_activities:
            if sub_activity['sub_activity_id'] == 'reaching': # TODO
            # if True:
                
                if sub_activity['end_frame'] - sub_activity['start_frame'] <= 6:
                    continue

                list.append((sub_activity['start_frame'], 
                                        sub_activity['end_frame']))
                tot_len += \
                    (sub_activity['end_frame'] - sub_activity['start_frame'])

# Extract the Sequences
pos_arr = np.concatenate([pos_arr[start:end, :] for (start, end) in list],
                        axis=0)
ori_arr = np.concatenate([ori_arr[start:end, :] for (start, end) in list],
                        axis=0)
list_new = []
cnt = 0
for (start, end) in list:
    list_new.append([cnt, cnt+end-start])
    cnt += (end-start)

index = np.array(list_new)

# From Y-up to Z-up Coordinate System
Ry2z=np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
pos_arr, ori_arr = change_space(pos_arr, ori_arr, R=Ry2z)

# Preprocess
print 'Preprocessing the joint of interests...'

ori_torso_arr = ori_arr[:, ori_ind(0)]

list_joi = [joints.index('left_hand'), joints.index('right_hand')]
data_joi, pos_joi_arr = preprocess_joi(list_joi, pos_arr, ori_torso_arr)
pos_joi_recon = postprocess_joi(data_joi)

print 'done.'

print 'Preprocessing the body-centered positions...'

bcpos_arr = preprocess_bcpos(skel, pos_arr, ori_torso_arr)
pos_recon_arr = postprocess_bcpos(skel, bcpos_arr)

print 'done.'

# Save and print
np.save(os.path.join(path_dataset, 'index'), index)
np.save(os.path.join(path_dataset, 'pos_arr'), pos_arr)
np.save(os.path.join(path_dataset, 'ori_arr'), ori_arr)
np.save(os.path.join(path_dataset, 'data_joi'), data_joi)
np.save(os.path.join(path_dataset, 'pos_joi_arr'), pos_joi_arr)
np.save(os.path.join(path_dataset, 'bcpos_arr'), bcpos_arr)

toc = clock()
print 'Done in {} secs.'.format(toc-tic)

print 'Reoncstruction error of joints of interest: {}'\
        .format(np.mean((pos_joi_arr - pos_joi_recon)**2))
print 'Reoncstruction error of relative position: {}'\
        .format(np.mean((pos_arr - pos_recon_arr)**2))
