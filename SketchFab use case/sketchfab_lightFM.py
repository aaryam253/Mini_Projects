import csv
import copy
import helpers
import itertools
import lightfm.evaluation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import requests
import scipy.sparse as sp
import seaborn as sns
import sys

from lightfm import LightFM
from scipy.special import expit
from sklearn.feature_extraction import DictVectorizer
from skopt import forest_minimize

sns.set_style('white')
sys.path.append('../')

df = pd.read_csv('../data/model_likes_anon.psv',
				  sep='|', quoting=csv.QUOTE_MINIMAL,
				  quotechar='\\')

df.drop_duplicates(inplace=True)
df.head()

# threshold data to only indivate users and models with min 5 likes
df = helpers.threshold_interactions_df(df, 'uid', 'mid', 5, 5)

# change dataframe into a likes matrix
# also, build index to ID mappers
likes, uid_to_idx, idx_to_uid,\
mid_to_idx, idx_to_mid = helpers.df_to_matrix(df, 'uid', 'mid')

likes

train, test, user_index = helpers.train_test_split(likes, 5, fraction=0.2)

eval_train = train.copy()
non_eval_users = list(set(range(train.shape[0])) - set(user_index))

eval_train = eval_train.tolil()
for u in non_eval_users:
	eval_train[u, :] = 0.0
eval_train = eval_train.tocsr()

# encode all the side information about the SketchFab models
sideinfo = pd.read.csv('../data/model_feats.psv',
						sep='|', quoting=csv.QUOTE_MINIMAL,
						quotechar='\\')
sideinfo.head()

# build list of dictionaries containing features and weights
# in same order as idx_to_mid prescribes
feat_dlist = [{} for _ in idx_to_mid]
for idx, row in sideinfo.iterrows():
	feat_key = '{}_{}'.format(row.type, str(row.value).lower())
	idx = mid_to_idx.get(row.mid)
	if idx is not None:
		feat_dlist[idx][feat_key] = 1

feat_dlist[0]

dv = DictVectorizer()
item_features = dv.fit_transform(feat_dlist)

item_features

def print_log(row, header=False, spacing=12):
	top = ''
	middle = ''
	bottom = ''
	for r in row:
		top += '+{}'.format('-'*spacing)
		if isinstance(r, str):
			middle += '| {0:^{1}}'.format(r, spacing-2)
		elif isinstance(r, int):
			middle += '| {0:^{1}}'.format(r, spacing-2)
		elif (isinstance(r, float) or isinstance(r, np.float32) or isinstance(r, np.float64)):
			middle += '| {0:^{1}.5f} '.format(r, spacing-2)
		bottom += '+{}'.format('='*spacing)
	top += '+'
	middle += '|'
	bottom += '+'
	if header:
		print(top)
		print(middle)
		print(bottom)
	else:
		print(middle)
		print(top)

# calculates the learning curve of the WARP loss function
def patk_learning_curve(model, train, test, eval_train,
						iterarray, user_features=None,
						item_features=None, k=5,
						**fit_params):
	old_epoch = 0
	train_patk = []
	test_patk = []
	headers = ['Epoch', 'train p@5', 'test p@5']
	print_log(headers, header=True)
	for epoch in iterarray:
		more = epoch - old_epoch
		model.fit_partial(train, user_features=user_features,
						  item_features=item_features,
						  epochs=more, **fit_params)
		this_test = lightfm.evaluation.precision_at_k(model, test, train_interactions=None, k=k)
		this_train = lightfm.evaluation.precision_at_k(model, eval_train, train_interactions=None, k=k)

		train_patk.append(np.mean(this_train))
		test_patk.append(np.mean(this_test))
		row = [epoch, train_patk[-1], test_patk[-1]]
		print_log(row)
	return model, train_patk, test_patk

# initialise the model and fit it to the training data
model = LightFM(loss='warp', random_state=2016)
model.fit(train, epochs=0);

iterarray = range(10, 110, 10)

model, train_patk, test_patk = patk_learning_curve(
	model, train, test, eval_train, iterarray, k=5, **{'num_threads': 1})

def plot_patk(iterarray, patk,
              title, k=5):
    plt.plot(iterarray, patk);
    plt.title(title, fontsize=20);
    plt.xlabel('Epochs', fontsize=24);
    plt.ylabel('p@{}'.format(k), fontsize=24);
    plt.xticks(fontsize=14);
    plt.yticks(fontsize=14);

# Plot train on left
ax = plt.subplot(1, 2, 1)
fig = ax.get_figure();
sns.despine(fig);
plot_patk(iterarray, train_patk,
         'Train', k=5)

# Plot test on right
ax = plt.subplot(1, 2, 2)
fig = ax.get_figure();
sns.despine(fig);
plot_patk(iterarray, test_patk,
         'Test', k=5)

plt.tight_layout();

# Optimise hyperparameters with scikit-optimise
def objective(params):
    # unpack
    epochs, learning_rate,\
    no_components, alpha = params
    
    user_alpha = alpha
    item_alpha = alpha
    model = LightFM(loss='warp',
                    random_state=2016,
                    learning_rate=learning_rate,
                    no_components=no_components,
                    user_alpha=user_alpha,
                    item_alpha=item_alpha)
    model.fit(train, epochs=epochs,
              num_threads=4, verbose=True)
    
    patks = lightfm.evaluation.precision_at_k(model, test,
                                              train_interactions=None,
                                              k=5, num_threads=4)
    mapatk = np.mean(patks)
    # Make negative because we want to _minimize_ objective
    out = -mapatk
    # Handle some weird numerical shit going on
    if np.abs(out + 1) < 0.01 or out < -1.0:
        return 0.0
    else:
        return out

space = [(1, 260), # epochs
         (10**-4, 1.0, 'log-uniform'), # learning_rate
         (20, 200), # no_components
         (10**-6, 10**-1, 'log-uniform'), # alpha
        ]

res_fm = forest_minimize(objective, space, n_calls=250,
                     random_state=0,
                     verbose=True)

print('Maximimum p@k found: {:6.5f}'.format(-res_fm.fun))
print('Optimal parameters:')
params = ['epochs', 'learning_rate', 'no_components', 'alpha']
for (p, x_) in zip(params, res_fm.x):
    print('{}: {}'.format(p, x_))

eye = sp.eye(item_features.shape[0], item_features.shape[0]).tocsr()
item_features_concat = sp.hstack((eye, item_features))
item_features_concat = item_features_concat.tocsr().astype(np.float32)

def objective_wsideinfo(params):
    # unpack
    epochs, learning_rate,\
    no_components, item_alpha,\
    scale = params
    
    user_alpha = item_alpha * scale
    model = LightFM(loss='warp',
                    random_state=2016,
                    learning_rate=learning_rate,
                    no_components=no_components,
                    user_alpha=user_alpha,
                    item_alpha=item_alpha)
    model.fit(train, epochs=epochs,
              item_features=item_features_concat,
              num_threads=4, verbose=True)
    
    patks = lightfm.evaluation.precision_at_k(model, test,
                                              item_features=item_features_concat,
                                              train_interactions=None,
                                              k=5, num_threads=3)
    mapatk = np.mean(patks)
    # Make negative because we want to _minimize_ objective
    out = -mapatk
    # Weird shit going on
    if np.abs(out + 1) < 0.01 or out < -1.0:
        return 0.0
    else:
        return out

space = [(1, 260), # epochs
         (10**-3, 1.0, 'log-uniform'), # learning_rate
         (20, 200), # no_components
         (10**-5, 10**-3, 'log-uniform'), # item_alpha
         (0.001, 1., 'log-uniform') # user_scaling
        ]
x0 = res_fm.x.append(1.)
# This typecast is required
item_features = item_features.astype(np.float32)
res_fm_itemfeat = forest_minimize(objective_wsideinfo, space, n_calls=50,
                                  x0=x0,
                                  random_state=0,
                                  verbose=True)

print('Maximimum p@k found: {:6.5f}'.format(-res_fm_itemfeat.fun))
print('Optimal parameters:')
params = ['epochs', 'learning_rate', 'no_components', 'item_alpha', 'scaling']
for (p, x_) in zip(params, res_fm_itemfeat.x):
    print('{}: {}'.format(p, x_))

# feature embeddings
epochs, learning_rate,\
no_components, item_alpha,\
scale = res_fm_itemfeat.x

user_alpha = item_alpha * scale
model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=learning_rate,
                no_components=no_components,
                user_alpha=user_alpha,
                item_alpha=item_alpha)
model.fit(likes, epochs=epochs,
          item_features=item_features_concat,
          num_threads=4)

# feature sorting
idx = dv.vocabulary_['tag_tiltbrush'] + item_features.shape[0]

def cosine_similarity(vec, mat):
    sim = vec.dot(mat.T)
    matnorm = np.linalg.norm(mat, axis=1)
    vecnorm = np.linalg.norm(vec)
    return np.squeeze(sim / matnorm / vecnorm)

tilt_vec = model.item_embeddings[[idx], :]
item_representations = item_features_concat.dot(model.item_embeddings)
sims = cosine_similarity(tilt_vec, item_representations)

def get_thumbnails(row, idx_to_mid, N=10):
    thumbs = []
    mids = []
    for x in np.argsort(-row)[:N]:
        response = requests.get('https://sketchfab.com/i/models/{}'\
                                .format(idx_to_mid[x])).json()
        thumb = [x['url'] for x in response['thumbnails']['images']
                 if x['width'] == 200 and x['height']==200]
        if not thumb:
            print('no thumbnail')
        else:
            thumb = thumb[0]
        thumbs.append(thumb)
        mids.append(idx_to_mid[x])
    return thumbs, mids











