import wadaken
import numpy as np
from sklearn import datasets

df = datasets.load_linnerud(as_frame=True).frame

if type(df) != np.ndarray:
    df = df.to_numpy()

x = df[:, 3].reshape(len(df))
X = np.array([np.ones((len(x))), x])

Y = df[:, 0].reshape(len(df))

ans = wadaken.akane(Y, X)

print(f'Beta hat: {ans[0]}')
print('-------------------------------------')
print(f'P value: {ans[7]}')
