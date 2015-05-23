import sys
sys.path.append('/deep/u/kuanfang/action-prediction');

from utils.io import *

path_data = '../CAD-120/Subject3_annotations'

# activity = read_activity(path_data)
# for sub_activity in activity:
#     print '++++++++++'
#     print sub_activity['id']
#     print sub_activity['activity_id']
#     print sub_activity['objects']

subject = read_subject(path_data)
print_subject(subject)
