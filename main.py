import statistics
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

# data = np.array([
#     185, 180, 180, 180, 178,
#     176, 176, 174, 174, 172,
#     169, 168, 167, 165, 164,
#     160, 159, 156, 156, 152,
#     151, 148, 145
# ])

# data = np.array([
#     11, 13, 20, 4, 8, 12, 0, 2, 0, 11, 17, 5, 20, 19, 0, 11, 10, 16, 8, 7, 6, 2, 13, 14, 4, 6, 16, 6, 19, 8, 20, 8, 12, 1, 1, 3, 10, 6, 19, 0, 16, 6, 10, 7, 17, 9, 16, 18, 1, 0, 5, 4, 13, 18, 4, 7, 14, 18, 3, 14, 3, 6, 3, 12, 18, 12, 11, 14, 10, 14, 20, 16, 18, 10, 14, 18, 2, 9, 13, 15, 16, 16, 5, 5, 6, 2, 18, 19, 15, 20, 20, 11, 15, 7, 8, 7, 3, 15, 7, 4, 19, 12, 19, 19, 18, 3, 16, 6, 8, 20, 3, 7, 15, 4, 17, 8, 2, 20, 6, 4, 14, 1, 20, 3, 5, 12, 16, 7, 15, 18, 4, 6, 4, 0, 14, 17, 8, 15, 19, 8, 4, 12, 0, 20, 20, 7, 2, 4, 18, 11, 6, 11, 8, 6, 5, 14, 10, 2, 5, 6, 7, 10, 12, 1, 10, 15, 0, 14, 20, 17, 8, 19, 8, 6, 16, 9, 19, 18, 16, 1, 11, 17, 12, 0, 6, 3, 7, 19, 15, 11, 0, 7, 6, 3, 13, 3, 19, 12, 8, 7, 3, 7, 2, 19, 20, 18, 11, 3, 4, 11, 17, 14, 8, 1, 6, 0, 9, 10, 9, 20, 9, 5, 0, 13, 2, 13, 10, 20, 12, 20, 16, 13, 20, 4, 4, 12, 1, 8, 17, 15, 3, 15, 8, 18, 11, 1, 18, 8, 15, 0, 0, 10, 8, 8, 15, 4, 10, 6, 2, 3, 12, 5, 3, 12, 1, 7, 6, 12, 1, 5, 11, 19, 3, 11, 15, 18, 2, 13, 18, 20, 2, 20, 19, 2, 0, 10, 16, 11, 19, 11, 3, 11, 19, 0, 2, 12, 0, 15, 19, 14
# ])

# data = np.array([
#     158, 146, 157, 149, 125, 144, 132, 158, 164, 138,
#     176, 138, 126, 168, 144, 158, 148, 136, 147, 140,
#     153, 135, 147, 142, 173, 146, 165, 154, 119, 163,
#     128, 145, 156, 150, 142, 145, 135, 161, 135, 140
# ])

# data = np.array([
#     13, 12, 11, 8, 7, 10, 13, 12, 12, 14,
#     10, 11, 12 ,13, 14, 10 ,10, 9, 13, 14,
#     14, 12 ,13, 13, 12, 12, 9, 14, 17, 12,
#     9, 10, 12, 11, 11, 12, 15, 15, 7, 9
# ])

data = np.array([
    57, 70, 37, 40, 52, 55, 60, 67, 40 ,48,
    41, 59, 48, 44, 38, 56, 57, 41, 64, 75,
    66, 46, 52, 40, 34, 50, 61, 57, 35, 53,
    45, 58, 43, 62, 57, 48, 52, 69, 53, 45,
    53, 58, 37, 63, 62, 57, 48, 51, 54, 46,
    59, 52, 62, 54, 49, 52, 69, 60, 32, 51
])     

data_size = data.size  # n

variation_range = np.max(data) - np.min(data)  # R
print(f'Variation Range: {variation_range}')

classification_number = math.floor(1 + (3.3 * math.log(data_size, 10)))  # K
print(f'Classification Number: {classification_number}')

classification_step = math.floor(variation_range / classification_number) + 1  # I
print(f'Classification Step: {classification_step}')

classification_steps = []

for stepIndex in range(classification_number):
    step_start_with = np.min(data) + (classification_step * stepIndex)
    step_end_with = (step_start_with + (classification_step - 1))
    steps_average = (step_start_with + step_end_with) / 2
    
    classification_steps.insert(
        0, [
            step_start_with,
            step_end_with,
            steps_average
        ]
    )
    
classification_steps = np.array(classification_steps)
classifRangeCountInData = []
howManyLessThanEndOfClassifExistInData = []

for classif in classification_steps:
    eachClassifRangeCountInData = np.bitwise_and(classif[1] >= data, data >= classif[0]).sum()
    classifRangeCountInData.append(eachClassifRangeCountInData)
    
    eachHowManyLessThanEndOfClassifExistInData = np.count_nonzero(classif[1] >= data)
    howManyLessThanEndOfClassifExistInData.append(eachHowManyLessThanEndOfClassifExistInData)

classification_steps = np.insert(classification_steps, 3, classifRangeCountInData, axis=1)
classification_steps = np.insert(classification_steps, 4, howManyLessThanEndOfClassifExistInData, axis=1)

xi = classification_steps[0::, 2]
fi = classification_steps[0::, 3]

classification_steps = np.insert(classification_steps, 5, fi * xi, axis=1)

dateFrame_columns = ['Cl_start', 'Cl_end', 'Xi', 'Fi', 'CFi', 'XiFi']
dataFrame = pd.DataFrame(classification_steps, columns=dateFrame_columns)

print(dataFrame)
XiFi = classification_steps[0::, 5]
print(f'XiFi total: {XiFi.sum()}')

x = np.vstack((xi, fi))

# plt.hist(x, color='#ac272e', ec='black')
# plt.hist(x)

fig, ax = plt.subplots()
ax.bar(x[0], x[1], width=5, linewidth=0.7)

# ax.set(xlim=(0, 20), xticks=np.arange(1, 20),
#        ylim=(0, 20), yticks=np.arange(1, 20))

plt.show()