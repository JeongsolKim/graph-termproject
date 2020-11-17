from preprocess import *
import time


t = time.time()
H1, H2, feats, label = load_graph(file_path = './train/train_800.txt')
print(time.time()-t)