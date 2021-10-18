import numpy as np
import matplotlib.pyplot as plt

print('now add one more line')

array_1 = np.array([[1, 3, 5],
                    [2, 4, 6],
                    [3, 6, 9],
                    [10, 11, 12]])
array_2 = array_1.flatten()
print(array_1, '\n', array_2)

array_3 = np.arange(48).reshape(4, 6, 2)
print('array_3:\n', array_3.shape)
array_4 = np.rollaxis(array_3, axis=2, start=0)
print('array_4:\n', array_4.shape)

array_5 = np.swapaxes(array_3, 0, 1)
print('array_5:\n', array_5.shape)

print(np.split(array_2, [3, 8]))
print(array_1.shape)

print(np.shape(array_1.reshape((3, 4))))
print(np.resize(array_1, (5, 4)))

# from matplotlib import pyplot as plt
# plt.hist(array_1.flatten(), bins=np.arange(13))
# plt.title('histogram')
# plt.show()

print(array_1, '\n', array_1.ravel())

Z = np.zeros((6, 6), dtype=int)
Z[1::2, ::2] = 1
Z[::2, 1::2] = 1
print(Z)

Z = np.random.rand(10, 10)
Z[np.random.randint(0, 10, size=15), np.random.randint(0, 10, size=15)] = np.nan
print(Z, '\n', np.sum(np.isnan(Z).sum()))
print('missing value locations:\n', np.argwhere(np.isnan(Z)))
print(np.isnan(Z))
Z[np.isnan(Z)] = 0
print(Z)

import pandas as pd
print(pd.__version__)

# series, including create, manipulate, delete, querry
arr = [0, 1, 2, 3, 4, 5]
s1 = pd.Series(arr)
print(s1)
order = [1, 2, 3, 4, 5, 6]
print(pd.Series(arr, index=order))

n = np.random.randn(5)
index = ['a', 'b', 'c', 'd', 'e']
s2 = pd.Series(n, index=index)
print(s2)

d = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
s3 = pd.Series(d)
print(s3)

s1.index = ['A', 'B', 'C', 'D', 'E', 'F']
print(s1)

print(s1[2:-1])

s4 = s1.append(s3)
print(s4)

print(s4.drop('e'))

# series operations
arr1 = np.arange(8)
arr2 = np.array([6, 7, 8, 9, 5])

s5 = pd.Series(arr2)
s6 = pd.Series(arr1)
print(s5)
print(s5.add(s6))
print(s5.sub(s6))
print(s5.mul(s6))
print(s5.div((s6)))
print('median:', s6.median(), 'max:', s6.max())

# dataframe
dates = pd.date_range('today', periods=6)
print(dates)
num_arr = np.random.randn(6, 4)
columns = ['A', 'B', 'C', 'D']
df1 = pd.DataFrame(num_arr, index=dates, columns=columns)
print(df1.tail(2))
print(df1.index, '\n', df1.columns, '\n', df1.describe())
print(df1.T)
print(df1.sort_values(by='A')[1:3])
# print(df1[:2])
# print(df1[-2:])
print(df1[['A', 'C']])
df2 = df1.copy()
df2.iloc[[1, 0]] = np.nan
print(df2)
# df2.iloc[1, 0, 3]['C'] = np.nan
print(df2.isnull())
df2.iloc[1]['A'] = 100
print(df2)
print(df2[['A', 'B']].max())

ts = pd.Series(np.random.randn(50), index = pd.date_range('today', periods=50))
ts = ts.cumsum()
# ts.plot()
# plt.show()

df = pd.DataFrame(np.random.randn(50, 4), index=ts.index, columns=['A', 'B', 'X', 'Y'])
df = df.cumsum()
# df.plot()
# plt.show()

df = pd.DataFrame({'A': [1, 2, 2, 2, 4, 4, 5, 5, 6, 6, 7, 8, 8]})
print(df.loc[df['A'].shift() != df['A']])
# print(df)

x = np.linspace(0, 10, 25)
y = x * x + 2

# plt.plot(x, y, 'r')
# plt.subplot(1, 2, 1)
# plt.plot(x, y, 'r--')
# plt.subplot(2, 2, 2)
# plt.plot(x, y, 'g*-')
# plt.show()

fig = plt.figure()
axes = fig.add_axes([.18, .1, .8, .8])
axes.plot(x, y, 'r')
# plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), dpi=100)
for ax in axes:
    ax.plot(x, y, 'r')

fig = plt.figure()
axes1 = fig.add_axes([.1, .1, .8, .8])
axes2 = fig.add_axes([.5, .5, .2, .2])
axes1.plot(x, y, 'r')
axes2.plot(y, x, 'g')

fig = plt.figure(figsize=(16, 9), dpi=100)
fig.add_subplot()
plt.plot(x, y, 'r')
plt.show()
