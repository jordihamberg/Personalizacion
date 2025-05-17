import math
import pandas as pd
import random
import numpy as np
import math
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def rating_average (ratings, NUM_ANIMES, u):
  acc = 0
  count = 0

  for i in range(NUM_ANIMES):
    if ratings[u][i] != None:
      acc += ratings[u][i]
      count += 1

  if count == 0:
    return None
  avg = acc / count

  return avg

def correlation_similarity (ratings, NUM_ANIMES, u, v):
  num = 0

  den_u = 0
  den_v = 0

  count = 0

  avg_u = rating_average(ratings, NUM_ANIMES, u)
  avg_v = rating_average(ratings, NUM_ANIMES, v)

  for i in range(NUM_ANIMES):
    if ratings[u][i] != None and ratings[v][i] != None:
      r_u = ratings[u][i]
      r_v = ratings[v][i]

      num += (r_u - avg_u) * (r_v - avg_v)
      den_u += (r_u - avg_u) * (r_u - avg_u)
      den_v += (r_v - avg_v) * (r_v - avg_v)

      count += 1

  if count > 0 and den_u != 0 and den_v != 0:
    cor = num / math.sqrt( den_u * den_v )
    return cor
  else:
    return None

def jmsd_similarity (ratings, NUM_ANIMES, MIN_RATING, MAX_RATING, u, v):

  union = 0
  intersection = 0
  diff = 0

  for i in range(NUM_ANIMES):
    if ratings[u][i] != None and ratings[v][i] != None:
      r_u = (ratings[u][i] - MIN_RATING) / (MAX_RATING - MIN_RATING)
      r_v = (ratings[v][i] - MIN_RATING) / (MAX_RATING - MIN_RATING)

      diff += (r_u - r_v) * (r_u - r_v)

      intersection += 1
      union += 1

    elif ratings[u][i] != None or ratings[v][i] != None:
      union += 1


  if intersection > 0:
    jaccard = intersection / union
    msd = diff / intersection
    return jaccard * (1 - msd)
  else:
    return None

def get_neighbors (k, similarities):

  neighbors = [None for _ in range(k)]

  for n in range(k):

    max_similarity = 0
    neighbor = None

    for v, sim in enumerate(similarities):
      if v not in neighbors and sim != None and sim > max_similarity:
        max_similarity = sim
        neighbor = v

    neighbors[n] = neighbor

  return neighbors

def average_prediction (ratings, i, neighbors):
  acc = 0
  count = 0

  for n in neighbors:
    if n == None: break
    if ratings[n][i] != None:
      acc += ratings[n][i]
      count += 1

  if count > 0:
    prediction = acc / count
    return prediction
  else:
    return None

def weighted_average_prediction (ratings, i, neighbors, similarities):
  num = 0
  den = 0

  for n in neighbors:
    if n == None: break

    if ratings[n][i] != None:
      num += similarities[n] * ratings[n][i]
      den += similarities[n]

  if den > 0:
    prediction = num / den
    return prediction
  else:
    return None

def deviation_from_mean_prediction (ratings, NUM_ANIMES, u, i, neighbors):
  acc = 0
  count = 0

  for n in neighbors:
    if n == None: break

    if ratings[n][i] != None:
      avg_n = rating_average(ratings, NUM_ANIMES, n)
      acc += ratings[n][i] - avg_n
      count += 1

  if count > 0:
    avg_u = rating_average(ratings, NUM_ANIMES, u)
    prediction = avg_u + acc / count
    return prediction
  else:
    return None

def get_recommendations (N, predictions):
  recommendations = [None for _ in range(N)]

  for n in range(N):

    max_value = 0
    anime = None

    for i, value in enumerate(predictions):
      if i not in recommendations and value != None and value > max_value:
        max_value = value
        anime = i

    recommendations[n] = anime

  return recommendations

def has_test_ratings (test_ratings, NUM_ANIMES, u):
  for i in range(NUM_ANIMES):
    if test_ratings[u][i] != None:
      return True
  return False

def get_user_mae (test_ratings, NUM_ANIMES, u, predictions):
  mae = 0
  count = 0

  for i in range(NUM_ANIMES):
    if test_ratings[u][i] != None and predictions[u][i] != None:
      mae += abs(test_ratings[u][i] - predictions[u][i])
      count += 1

  if count > 0:
    return mae / count
  else:
    return None

def get_mae (test_ratings, NUM_ANIMES, NUM_USERS, predictions):
  mae = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings (test_ratings, NUM_ANIMES, u):
      user_mae = get_user_mae(test_ratings, NUM_ANIMES, u, predictions)

      if user_mae != None:
        mae += user_mae
        count += 1


  if count > 0:
    return mae / count
  else:
    return None

def get_user_rmse (test_ratings, NUM_ANIMES, u, predictions):
  mse = 0
  count = 0

  for i in range(NUM_ANIMES):
    if test_ratings[u][i] != None and predictions[u][i] != None:
      mse += (test_ratings[u][i] - predictions[u][i]) * (test_ratings[u][i] - predictions[u][i])
      count += 1

  if count > 0:
    return math.sqrt(mse / count)
  else:
    return None

def get_rmse (test_ratings, NUM_ANIMES, NUM_USERS, predictions):
  rmse = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings (test_ratings, NUM_ANIMES, u):
      user_rmse = get_user_rmse(test_ratings, NUM_ANIMES, u, predictions)

      if user_rmse != None:
        rmse += user_rmse
        count += 1


  if count > 0:
    return rmse / count
  else:
    return None

def get_user_precision (test_ratings, theta, u, predictions, N):
  precision = 0
  count = 0
  recommendations = get_recommendations(N, predictions[u])

  for i in recommendations:
    if i != None and test_ratings[u][i] != None:
      precision += 1 if test_ratings[u][i] >= theta else 0
      count += 1

  if count > 0:
    return precision / count
  else:
    return None

def get_precision(test_ratings, NUM_ANIMES, NUM_USERS, theta, predictions, N):
  precision = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings (test_ratings, NUM_ANIMES, u):
      user_precision = get_user_precision(test_ratings, theta, u, predictions, N)

      if user_precision != None:
        precision += user_precision
        count += 1


  if count > 0:
    return precision / count
  else:
    return None

def get_user_recall (test_ratings, NUM_ANIMES, theta, u, predictions, N):
  recall = 0
  count = 0
  recommendations = get_recommendations(N, predictions[u])

  for i in range(NUM_ANIMES):
    if test_ratings[u][i] != None and predictions[u][i] != None:
      if test_ratings[u][i] >= theta:
        recall += 1 if i in recommendations else 0
        count += 1

  if count > 0:
    return recall / count
  else:
    return None

def get_recall (test_ratings, NUM_ANIMES, NUM_USERS,theta, predictions, N):
  recall = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings(test_ratings, NUM_ANIMES, u):
      user_recall = get_user_recall(test_ratings, NUM_ANIMES, theta, u, predictions, N)

      if user_recall != None:
        recall += user_recall
        count += 1


  if count > 0:
    return recall / count
  else:
    return None

def get_user_f1 (test_ratings, NUM_ANIMES, theta, u, predictions, N):
  precision = get_user_precision(test_ratings, theta, u, predictions, N)
  recall = get_user_recall(test_ratings, NUM_ANIMES, theta, u, predictions, N)

  if precision == None or recall == None:
    return None
  elif precision == 0 and recall == 0:
    return 0
  else:
    return 2 * precision * recall / (precision + recall)
  
  
def get_f1 (test_ratings, NUM_ANIMES, NUM_USERS, predictions, theta, N):
  f1 = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings (test_ratings, NUM_ANIMES, u):
      user_f1 = get_user_f1(test_ratings, NUM_ANIMES, theta, u, predictions, N)

      if user_f1 != None:
        f1 += user_f1
        count += 1


  if count > 0:
    return f1 / count
  else:
    return None

def get_ordered_test_animes(test_ratings, u):
  num_animes = sum(x is not None for x in test_ratings[u])
  animes = [None for _ in range(num_animes)]

  for n in range(num_animes):

    max_value = 0
    anime = None

    for i,value in enumerate(test_ratings[u]):
      if i not in animes and value != None and value > max_value:
        max_value = value
        anime = i

    animes[n] = anime

  return animes

def get_user_idcg (test_ratings, u):
  animes = get_ordered_test_animes(test_ratings, u)
  idcg = 0

  for pos, i in enumerate(animes):
    idcg += (2 ** test_ratings[u][i] - 1) / math.log(pos+2, 2)

  return idcg

def get_user_dcg (test_ratings, u, recommendations):
  dcg = 0

  for pos, i in enumerate(recommendations):
    if i != None and test_ratings[u][i] != None:
      dcg += (2 ** test_ratings[u][i] - 1) / math.log(pos+2, 2)

  return dcg

def get_user_ndcg (test_ratings, u, predictions, N):
  recommendations = get_recommendations(N, predictions[u])
  dcg = get_user_dcg(test_ratings, u, recommendations)
  idcg = get_user_idcg(u)
  if idcg == 0:
    return 0
  else:
    return dcg / idcg
  
def get_ndcg (test_ratings, NUM_ANIMES, NUM_USERS, predictions, N):
  ndcg = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings (test_ratings, NUM_ANIMES, u):
      user_ndcg = get_user_ndcg(test_ratings, u, predictions, N)

      if user_ndcg != None:
        ndcg += user_ndcg
        count += 1


  if count > 0:
    return ndcg / count
  else:
    return None

def preprocesar_dataframe_animes(dataframe_path,
                                 n_user_ratings, n_anime_ratings,
                                 num_users=1000,
                                 test_size = 0.2, RANDOM_STATE = 42):
    
    # Hacemos el dataframe con el csv.
    df_ratings = pd.read_csv(dataframe_path, encoding='utf8')

    # Filtramos ratings con -1
    df_ratings = df_ratings[df_ratings['rating'] != -1]    
    
    # Filtramos usuarios con menos de n_user_ratings ratings y
    # filtramos animes con menos de n_anime_ratings ratings.
    usuarios_utiles = df_ratings['user_id'].value_counts()[df_ratings['user_id'].value_counts() > n_user_ratings].index
    df_ratings = df_ratings[df_ratings['user_id'].isin(usuarios_utiles)]
    animes_utiles = df_ratings['anime_id'].value_counts()[df_ratings['anime_id'].value_counts() > n_anime_ratings].index
    df_ratings = df_ratings[df_ratings['anime_id'].isin(animes_utiles)]
    
    # Muestreamos a la cantidad de usuarios que queramos y volvemos a filtrar pelis.
    usuarios_unicos = df_ratings['user_id'].unique()
    sampled_user_ids = np.random.choice(usuarios_unicos, size=num_users, replace=False)
    df_ratings = df_ratings[df_ratings['user_id'].isin(sampled_user_ids)]
    
    animes_utiles = df_ratings['anime_id'].value_counts()[df_ratings['anime_id'].value_counts() > n_anime_ratings//2].index
    df_ratings = df_ratings[df_ratings['anime_id'].isin(animes_utiles)]
    
    # Factorizamos.
    df_ratings['user_id'], user_codes = pd.factorize(df_ratings['user_id'])
    df_ratings['anime_id'], anime_codes = pd.factorize(df_ratings['anime_id'])
    df_ratings.reset_index(drop=True, inplace=True)
    
    # Sacamos métricas globales.
    NUM_USERS = len(user_codes)
    NUM_ANIMES = len(anime_codes)
    MIN_RATING = df_ratings['rating'].min()
    MAX_RATING = df_ratings['rating'].max()
    SCORES = sorted(df_ratings.rating.unique())
    
    # Dividimos en Train test
    train_df, test_df = train_test_split(df_ratings,
                                     test_size=test_size,          
                                     random_state=RANDOM_STATE,    
                                     stratify=df_ratings['user_id'])
    train_user_ids = train_df['user_id'].unique()
    train_anime_ids = train_df['anime_id'].unique()
    test_df = test_df[test_df['user_id'].isin(train_user_ids)]
    test_df = test_df[test_df['anime_id'].isin(train_anime_ids)]
    
    # Hacemos las matrices
    ratings_train_matrix = [[None for _ in range(NUM_ANIMES)] for _ in range(NUM_USERS)]
    for _, row in train_df.iterrows():
        ratings_train_matrix[int(row.user_id)][int(row.anime_id)] = int(row.rating)
    
    ratings_test_matrix = [[None for _ in range(NUM_ANIMES)] for _ in range(NUM_USERS)]
    for _, row in test_df.iterrows():
        ratings_test_matrix[int(row.user_id)][int(row.anime_id)] = int(row.rating)
    
    return (train_df, test_df, 
            ratings_train_matrix, ratings_test_matrix,
            NUM_USERS, NUM_ANIMES, MIN_RATING, MAX_RATING, SCORES)
    
def get_metricas(test_ratings, NUM_ANIMES, NUM_USERS, predictions, theta, N):
  print("MAE = ", get_mae(test_ratings, NUM_ANIMES, NUM_USERS, predictions))
  print("RMSE = ", get_rmse(test_ratings, NUM_ANIMES, NUM_USERS, predictions))
  print("Precision = ", get_precision(test_ratings, NUM_ANIMES, NUM_USERS, theta, predictions, N))
  print("Recall = ", get_recall(test_ratings, NUM_ANIMES, NUM_USERS,theta, predictions, N))
  print("F1 = ", get_f1(test_ratings, NUM_ANIMES, NUM_USERS, predictions, theta, N))