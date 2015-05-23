# hyper parameters of CAD-120 skeleton data format

"""
Below code used to compute the indexes of the CAD-120 data format.
We could compute the indexs only once, remeber the indexes and abandon the code.
"""
# idx_ori = []#[None] * (9*11)
# idx_pos = []#[None] * (3*15)
# idx_ori_conf =  []#[None] * (11)
# idx_pos_conf =  []#[None] * (15)
# 
# for i in range(11):
#     start = 1 + i*(9+1+3+1)
#     for j in range(9):
#         idx_ori.append(start + j)
#     idx_ori_conf.append(start + 9)
#     for j in range(3):
#         idx_pos.append(start + 9 + 1 + j)
#     idx_pos_conf.append(start + 9 + 1 + 3)
# 
# for i in range(11, 15):
#     start = 1 + 11*(9+1+3+1) + (i-11)*(3+1)
#     for j in range(3):
#         idx_pos.append(start + j)
#     idx_pos_conf.append(start + 3)
# 
# print idx_ori
# print idx_pos
# print idx_ori_conf
# print idx_pos_conf

idx_ori = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            57, 58, 59, 60, 61, 62, 63, 64, 65, 71, 72, 73, 74, 75, 76, 77, 78,
            79, 85, 86, 87, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103, 104,
            105, 106, 107, 113, 114, 115, 116, 117, 118, 119, 120, 121, 127, 
            128, 129, 130, 131, 132, 133, 134, 135, 141, 142, 143, 144, 145, 
            146, 147, 148, 149]
idx_pos = [11, 12, 13, 25, 26, 27, 39, 40, 41, 53, 54, 55, 67, 68, 69, 81, 82, 
            83, 95, 96, 97, 109, 110, 111, 123, 124, 125, 137, 138, 139, 151, 
            152, 153, 155, 156, 157, 159, 160, 161, 163, 164, 165, 167, 168, 
            169] 
idx_ori_conf = [10, 24, 38, 52, 66, 80, 94, 108, 122, 136, 150]
idx_pos_conf = [14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 158, 162, 166,
                170]

dim_ori = 9
dim_pos = 3
num_ori = len(idx_ori) / dim_ori
num_pos = len(idx_pos) / dim_pos


# connect relationship
"""
HEAD - NECK
LEFT_HIP - RIGHT_HIP


NECK - LEFT_SHOULDER
LEFT_SHOULDER - TORSOR
TORSOR - LEFT_HIP

LEFT_SHOULDER - LEFT_ELBOW
LEFT_ELBOW - LEFT_HAND

LEFT_HIP - LEFT_KNEE
LEFT_KNEE - LEFT_FOOT


NECK - RIGHT_SHOULDER
RIGHT_SHOULDER - TORSOR
TORSOR - RIGHT_HIP

RIGHT_SHOULDER - RIGHT_ELBOW
RIGHT_ELBOW - RIGHT_HAND

RIGHT_HIP - RIGHT_KNEE
RIGHT_KNEE - RIGHT_FOOT
"""

"""
1 - 2
8 - 10


2 - 4
4 - 3
3 - 8

4 - 5
5 - 12

8 - 9
9 - 14


2 - 6
6 - 3
3 - 10

6 - 7
7 - 13

10 - 11
11 - 15
"""

# 1-based index
connect_1based = [(1, 2), (8, 10), 
                    (2, 4), (4, 3), (3, 8),
                    (4, 5), (5, 12), (8, 9), (9, 14),
                    (2, 6), (6, 3), (3, 10),
                    (6, 7), (7, 13), (10, 11), (11, 15)]

# 0-based index
connect = [(a-1, b-1) for a, b in connect_1based]

