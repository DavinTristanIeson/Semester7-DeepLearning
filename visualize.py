import pandas as pd
import retina
import matplotlib.pyplot as plt
import sklearn.manifold

df = pd.read_csv(retina.filesys.TRAINING_DATA_CSV_PATH, index_col=False)
labels = df["label"]
df = df.drop(columns=["label"])

pca = sklearn.manifold.TSNE(n_components=2)
points = pca.fit_transform(df)

plt.scatter(points[:,0], points[:,1], c=labels) # type: ignore
plt.legend()
plt.show()


