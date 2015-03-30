import numpy as np
from scipy import stats as s
from scipy.stats import multivariate_normal as mn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

H = np.loadtxt('/home/user/PRTakeHome1/train_sp2015_v14')
St= np.loadtxt('/home/user/PRTakeHome1/test_sp2015_v14')

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(H)
distances, indices = nbrs.kneighbors(St)

f = open('/home/user/Desktop/TestClassification_1KNN','w')
for i in range(15000):
    if( indices[i][0]>=0 and indices[i][0]<5000):
        f.write('1')
        f.write('\n')
    elif( indices[i][0]>=5000 and indices[i][0]<10000 ):
        f.write('2')
        f.write('\n')
    else:
        f.write('3')
        f.write('\n')
f.close()

f = open('/home/user/Desktop/TestClassification_3KNN','w')
for i in range(15000):
    freq = [0, 0, 0]
    for j in range(3):
        if( indices[i][j]>=0 and indices[i][j]<5000 ):
            freq[0] = freq[0] + 1
        elif( indices[i][j]>=5000 and indices[i][j]<10000 ):
            freq[1] = freq[1] + 1
        else:
            freq[2] = freq[2] + 1
    val = freq.index(max(freq)) + 1
    f.write( str(val) + '\n' )
        
f.close()

f = open('/home/user/Desktop/TestClassification_5KNN','w')
for i in range(15000):
    freq = [0, 0, 0]
    for j in range(5):
        if( indices[i][j]>=0 and indices[i][j]<5000 ):
            freq[0] = freq[0] + 1
        elif( indices[i][j]>=5000 and indices[i][j]<10000 ):
            freq[1] = freq[1] + 1
        else:
            freq[2] = freq[2] + 1
    val = freq.index(max(freq)) + 1
    f.write( str(val) + '\n' )
        
f.close()

arr = [2,3,1,3,1,2]
err = 0
cof_mat = np.zeros((3,3))
for i in range(0, 15000):
    freq = [0, 0, 0]
    for j in range(5):
        if( indices[i][j]>=0 and indices[i][j]<5000 ):
            freq[0] = freq[0] + 1
        elif( indices[i][j]>=5000 and indices[i][j]<10000 ):
            freq[1] = freq[1] + 1
        else:
            freq[2] = freq[2] + 1
    retrievedRes = freq.index(max(freq)) + 1
    if(  retrievedRes != arr[i%6]):
        err = err + 1
    cof_mat[arr[i%6]-1][retrievedRes-1] = cof_mat[arr[i%6]-1][retrievedRes-1] + 1
print err
print cof_mat

arr = [2,3,1,3,1,2]
err = 0
cof_mat = np.zeros((3,3))
for i in range(0, 15000):
    freq = [0, 0, 0]
    for j in range(3):
        if( indices[i][j]>=0 and indices[i][j]<5000 ):
            freq[0] = freq[0] + 1
        elif( indices[i][j]>=5000 and indices[i][j]<10000 ):
            freq[1] = freq[1] + 1
        else:
            freq[2] = freq[2] + 1
    retrievedRes = freq.index(max(freq)) + 1
    if(  retrievedRes != arr[i%6]):
        err = err + 1
    cof_mat[arr[i%6]-1][retrievedRes-1] = cof_mat[arr[i%6]-1][retrievedRes-1] + 1
print err
print cof_mat

arr = [2,3,1,3,1,2]
err = 0
cof_mat = np.zeros((3,3))
for i in range(0, 15000):
    freq = [0, 0, 0]
    for j in range(1):
        if( indices[i][j]>=0 and indices[i][j]<5000 ):
            freq[0] = freq[0] + 1
        elif( indices[i][j]>=5000 and indices[i][j]<10000 ):
            freq[1] = freq[1] + 1
        else:
            freq[2] = freq[2] + 1
    retrievedRes = freq.index(max(freq)) + 1
    if(  retrievedRes != arr[i%6]):
        err = err + 1
    cof_mat[arr[i%6]-1][retrievedRes-1] = cof_mat[arr[i%6]-1][retrievedRes-1] + 1
print err
print cof_mat


