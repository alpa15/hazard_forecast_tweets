## LIBRARIES
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score


def read_data(path):
    """Function to open the .csv file and upload it as dataframe"""

    return pd.read_csv(path)


def data_preprocessing(df, nltk_data_path, path_file, save = False):
    """
    Transform data in a usable form.

    This function takes the dataframe as input and returns the preprocessed version.

    Parameters:
    df : pd.DataFrame
        The dataframe read from the original path.
    nltk_data_path : path
        Path where the Natural Language Toolkit library data can be found.
    path_file : path
        Path of the data.
    save : bool
        Boolean variable which states if the dataframe preprocessed must be saved.

    Returns:
    df : pd.DataFrame
        The dataframe in a refined form.

    Raises:
    TypeError: If the input is not a dataframe.
    
    Example:
    >>> data_preparation(df, nltk_data_path, path_file, True)
    df
    """

    # Error raise if the input is not a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input is not a dataframe")

    # Addition of the row "text_vect" to insert the vectorial representation
    # of every sentence
    df['text_vect'] = None

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Get the list of English stop words
    stop_words = set(stopwords.words('english'))

    # Load of the useful tools
    nltk.data.path.append(nltk_data_path)
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)
    nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)

    for row in df.values:
        # The sentence of the row is taken and tokenized
        idx_column_text = int(np.where(df.columns == 'text')[0][0])
        tokens = word_tokenize(row[idx_column_text])

        # Function to convert part of speech (POS) tags to WordNet's format
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return 'a'  # Adjective
            elif treebank_tag.startswith('V'):
                return 'v'  # Verb
            elif treebank_tag.startswith('N'):
                return 'n'  # Noun
            elif treebank_tag.startswith('R'):
                return 'r'  # Adverb
            else:
                return None  # If no match, return None

        # Lemmatize each token with its POS tag
        lemmatized_tokens = []
        for token, pos in pos_tag(tokens):
            wordnet_pos = get_wordnet_pos(pos) or 'n'  # Default to noun if no POS tag found
            lemmatized_token = lemmatizer.lemmatize(token, pos=wordnet_pos)
            lemmatized_tokens.append(lemmatized_token)

        # Join the lemmatized tokens back into a sentence
        lemmatized_sentence = ' '.join(lemmatized_tokens)

        filtered_tokens = [word for word in lemmatized_sentence.split()
                           if word.lower() not in stop_words]

        df.at[row[0], 'text_vect'] = filtered_tokens

    # If needed, save the dataframe to have it updated
    if save:
        df.to_csv(path_file, index=False)

    return df


def data_analysis(df):
    """
    Displays stats and analysis over the processed dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe where all the processed elements are contained.
    """
    print("\n------------------\nDATA ANALYSIS\n------------------")
    print("The keywords are:")
    print(df['keyword'].value_counts())
    print("\nThe percentage of real dangerous situations is: " +
          str(np.round(100 * np.sum(df['target']) / len(df), 1)) + " %")
    

def sparse_data_generation(df, mf = 100, disp = False):
    """
    Function which transforms the "text" column of the dataframe to a sparse
    vector which indicates the words present into every tweet

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe where all the processed elements are contained.
    mf : int
        The number of words with greatest presence in tweets, used as number
        of training variables.
    disp : bool
        Boolean variable which defines if the data must be shown or not.

    Returns:
    vectors : sparse matrix numpy.int64
        The sparse vector of all the words contained in every tweet.

    Raises:
    TypeError: If the input is not a dataframe.
    """

    # Error raise if the input is not a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input is not a dataframe")
    
    # If the dataframe is empty, it is returned an array with a 0 as single element
    if len(df) == 0:
        return np.array(0)

    # Initialization of the count vectorizer object
    count_vectorizer = feature_extraction.text.CountVectorizer(max_features=mf)

    # Every tweet is transformed into a dummy array to indicate which words are contained
    vectors = count_vectorizer.fit_transform(df["text"])

    if disp:
        print(vectors)
        
    return vectors


def train_val_test_data_split(df, train_prc, val_prc, clustering = True, k = 3):
    """
    Function which splits the dataset into training, validation and test
    Clustering operation can be applied, to have a dataset with a more uniform division of elements
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe where all the processed elements are contained.
    train_prc : float
        Value between 0 and 1, which defines the percentage of training data.
    val_prc : float
        Value between 0 and 1, which defines the percentage of validation data.
    clustering : bool
        Boolean variable which defines if the clustering should be applied or not.
    k : int
        Integer value which defines the k of the k-means algorithm.

    Returns:
    train_idx, val_idx, test_idx : (list, list, list)
        The 3 lists which contain the indexes of training, validation and test data.

    Raises:
    TypeError: If the input is not a dataframe.
    
    Example:
    >>> data_preparation(df, 0.7, 0.2, False, 3)
    [3, 65, 143, ...], [0, 2, ...], [1, 47, 102, ...]
    """

    # Error raise if the input is not a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input is not a dataframe")
    
    if train_prc + val_prc >= 1:
        raise ValueError("The percentages are not correct")

    if clustering:
        # Preprocess and vectorize the text data
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['text'])

        # Apply K-Means clustering with specified k
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=5, max_iter=500)
        kmeans.fit(X)

        # Get the cluster labels and add them to the dataframe
        labels = kmeans.labels_
        df['cluster'] = labels

        # The 3 different dataset are created
        train_idx = []
        val_idx = []
        test_idx = []

        # Loop to insert the elements of the different clusters into
        # the different dataframes, with the percentages defined as input
        for i in range(0, k):
            # All the rows belonging to the cluster 'i' are selected
            cluster_i = df[df['cluster'] == i]
            # The new elements are concatenated to the already existent
            if train_prc > 0:
                train_idx.extend(cluster_i[
                    :(int(train_prc*len(cluster_i)))]['id'].tolist())
            if val_prc > 0:
                val_idx.extend(cluster_i[
                    int(train_prc*len(cluster_i)):
                    int((train_prc+val_prc)*len(cluster_i))
                    ]['id'].tolist())
            if 1-train_prc-val_prc > 0:
                test_idx.extend(cluster_i[
                    int((train_prc+val_prc)*len(cluster_i)):]['id'].tolist())
    else:
        # Data are not grouped with clustering, but simply divided into the 3 splits
        train_idx = df.loc[0:int(len(df)*train_prc)]['idx'].tolist()
        val_idx = df.loc[int(len(df)*train_prc):
                         int(len(df)*train_prc)+int(len(df)*val_prc)]['idx'].tolist()
        test_idx = df.loc[int(len(df)*train_prc)+int(len(df)*val_prc):len(df)
                          ]['idx'].tolist()

    return train_idx, val_idx, test_idx


def train_random_forest(vectors, targets):
    """
    Function to train a random forest starting from the training set.

    Parameters
    ----------
    vectors : sparse matrix numpy.int64
        The sparse vector with all the words contained in every tweet.
    targets : list
        The target which determines if the tweet is dangerous or not.

    Returns
    -------
    model : 
        The trained model to be used with tweets.
    """
    
    # Initialize and fit the random forest classifier
    model = RandomForestClassifier()
    model = model.fit(vectors, targets)
    
    return model


def train_XGBoost(vectors, targets):
    """
    Function to train a XGBoost starting from the training set.

    Parameters
    ----------
    vectors : sparse matrix numpy.int64
        The sparse vector with all the words contained in every tweet.
    targets: list
        The target which determines if the tweet is dangerous or not.

    Returns
    -------
    model : 
        The trained model to be used with tweets.
    """

    # Initialize and fit the XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(vectors, targets)

    return model


def metrics(model, vectors_test, targets_test):
    """
    Function which prints the results of accuracy and F1 score, adding the
    information related to a classification report, to understand the
    performance of the model trained over the test data.

    Parameters
    ----------
    model:
        The model obtained after the training of the random forest or XGBoost.
    
    Example:
    >>> metrics(model, vectors_test, targets_test)
    Accuracy:  78.96 %
    F1 score:  25.52 %

    Classification Report:
                precision    recall  f1-score   support

            0       0.81      0.96      0.88      2682
            1       0.53      0.17      0.26       731

    accuracy                            0.79      3413
    macro avg       0.67      0.56      0.57      3413
    weighted avg    0.75      0.79      0.74      3413
    """

    predictions = model.predict(vectors_test)
    print("Accuracy: ", np.round(100 * accuracy_score(targets_test, predictions), 2), "%")
    print("F1 score: ", np.round(100 * f1_score(targets_test, predictions), 2), "%")

    report = classification_report(targets_test, predictions)
    print("\nClassification Report:\n", report)