from functions import *

# Loading of the Natural Language Toolkit resources to execute the data analysis
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Data are open from the path passed as input and the first lines are shown
path_csv_data = "tweets.csv"
df = read_data(path_csv_data)
print("\n-----------\nOriginal CSV file\n-----------")
print(df.head())

# Turn the "data_prepare" bool variable to True if "data_preprocessing" must be run
data_prepare = False
if data_prepare:
      nltk_data_path = "path\\to\\nltk_data"
      path_file = "path\\to\\file"
      save = True
      df = data_preprocessing(df, nltk_data_path, path_file, save)
else:
      path = "path\\to\\file"
      df = read_data(path)

print("\n-----------\nCSV file after preprocessing\n-----------")
print(df.head())

# The data analysis process is performed, showing the main stats
data_analysis(df)

# The number of the most used words used to be searched in every tweet
features_no = 100
words_in_tweets = sparse_data_generation(df, features_no)

train_split = 0.7
val_split = 0
clustering = True
cluster_no = 5
train_idx, val_idx, test_idx = train_val_test_data_split(
      df, train_split, val_split, clustering, cluster_no)

model_rf = train_random_forest(words_in_tweets[train_idx],
                               df.loc[train_idx]['target'].tolist())

metrics(model_rf, words_in_tweets[test_idx], df.loc[test_idx]['target'].tolist())

model_xgb = train_XGBoost(words_in_tweets[train_idx],
                               df.loc[train_idx]['target'].tolist())

metrics(model_xgb, words_in_tweets[test_idx], df.loc[test_idx]['target'].tolist())