# Sentiment analysis using POS-tagging & Sentiment Polarity computation
 
POS tagging is a necessary step for performing sentiment analysis, as the part of speech has a great impact on a word’s sentiment polarity. Designing an algorithm in which different parts of speech are assigned different sentiment weights. 
For example, we assume that adjectives convey the stronger sentiment information than verbs and nouns. So we assign larger sentiment weights to the adjectives. 
Verbs and nouns may also convey sentiment information from time to time. For example, the verb love and the noun congratulations are often associated with positive sentiment. 
However, to express the sentiment, we believe adjectives play a much more dominant role than verbs and nouns. Therefore, we will assign smaller sentiment weights to verbs and nouns than adjectives. S
imilarly, we should assign smaller or zero sentiment weights to determiner and preposition.

The data can be seen on [amazon_cells_labelled.csv](https://github.com/ricocahyadi777/pos-tagging-sentiment-analysis-nlp/blob/main/amazon_cells_labelled.csv). <br/>
I forget where do i get the data from. 

## Basic Naive Bayes
```python
def make_to_list(dataframe):    # Convert dataframe into list
    documents = []
    for index, row in dataframe.iterrows():
        tokens = nltk.word_tokenize(row['Comment']) # Tokenize each document
        documents.append((tokens, row['Label']))    # Add the tokenized document into the list, paired with the label
    return documents

def feature_extraction(document, word_features):
    stop_words = set(stopwords.words('english'))    # Create the list of stop words
    filtered_sentence = [w for w in document if not w in stop_words] # Filter the sentence by removing stop words
    document_words = set(filtered_sentence)     # Remove repeating word
    features = {}
    for word in word_features:  # Loop through all the word features
        # Enter the feature (Extract the feature from each document) to the dictionary
        features['contains({})'.format(word)] = (word in document_words)
    return features

def feature_selection(documents):
    word_list = []
    for docs, label in documents:   # Loop through all documents
        word_list.extend(docs)  # Append all the word into the list
    document_set = set(word_list)   # Remove repeating word
    # Get the distribution of the word
    all_words = nltk.FreqDist(word.lower()for word in document_set)
    # Get the top 2000 most frequent words as feature
    word_features = list(all_words)[:2000]
    return word_features
```
We make a few functions to help with 'making the data into a list', 'feature extraction', & 'feature selection'. <br/>
From here we just need to proceed with the steps using the function prepared accordingly.

```python
dataframe = pd.read_csv('amazon_cells_labelled.csv')    # Read the file as dataframe
docs = make_to_list(dataframe)  # Get the dataframe as a list

feature = feature_selection(docs)   #Select the feature
featuresets = [(feature_extraction(d, feature), c) for (d, c) in docs] # Extract the selected features from documents
train_set, test_set = featuresets[200:], featuresets[:200]  # Split test and train data

classifier = nltk.NaiveBayesClassifier.train(train_set) # Train the naive bayes classifier

# Checking the result
print("Accuracy is : {}%".format(accuracy(classifier, test_set)*100)) 
print(classifier.show_most_informative_features(10))  # Get 10 of the most informative features in the classifier
```

From Basic Naive Bayes we got a pretty solid result.
![image](https://github.com/ricocahyadi777/pos-tagging-sentiment-analysis-nlp/assets/63791918/30fcd27d-a05d-493e-b89f-b3e36648e98b)

Then we want to try adding POS-tagging into the algorithm for improved accuracy.

## POS-Tagging
First of all, we need to define a library for POS-tagging
```python
# Dictionary for nltk pos tag groupings
POS_TAG_GROUPINGS = {
    "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective",
    "NN": "Noun", "NNS": "Noun", "NNP": "Noun", "NNPS": "Noun",
    "PRP": "Pronoun", "PRP$": "Pronoun",
    "RB": "Adverb", "RBR": "Adverb", "RBS": "Adverb",
    "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb", "VBP": "Verb", "VBZ": "Verb",
    "VH": "Verb", "VHD": "Verb", "VHG": "Verb", "VHN": "Verb", "VHP": "Verb", "VHZ": "Verb",
    "VV": "Verb", "VVD": "Verb", "VVG": "Verb", "VVN": "Verb", "VVP": "Verb", "VVZ": "Verb",
}
```
To apply POS tagging for knowledge or lexicon extraction, we need to tag each word with the POS tag. 
The whole process will be:
1. Tag every word with the POS tag from nltk. 
2. Group the tag to Verb, Adjective, Noun, Pronoun, Adverb, or others. So, same word with different tag are treated differently.
3. Treat them as a pair.
4. Remove the repeated word tag pair.
5. Remove all the stop words from the list of features. 
6. Take the n-most frequent pair as the feature (In this case we use n = 2000)
7. The feature extracted will be whether the word tag pair exist in the document

Functions below are created to help us.

```python
def make_to_list(dataframe): # Convert dataframe into list
    documents = []
    for index, row in dataframe.iterrows():
        word_tokens = word_tokenize(row['Comment']) # Tokenize each document
        tags = pos_tag(word_tokens) # Get the tag for each tokenized word
        word_set = []
        for word, tag in zip(word_tokens, tags):    # Loop through the tokens and tag pair
            if tag[1] not in POS_TAG_GROUPINGS.keys():  # If the tag is not in the group we consider
                # Append the word in a tuple together with the tag as 'Others'
                word_set.append(tuple([word.lower(), 'Others']))
            else:   # If the tag is in the group we consider
                # Append the word in a tuple together with the tag group
                word_set.append(tuple([word.lower(), POS_TAG_GROUPINGS[tag[1]]]))
        # Append the whole set of row (document) into the main documents list
        documents.append((word_set, row['Label']))
    return documents

def feature_extraction(document, word_features):    # Get the feature for the data row (document)
    stop_words = set(stopwords.words('english'))    # Create the list of stop words
    filtered_sentence = [(w,t) for w, t in document if not w in stop_words] # Filter the sentence by removing stop words
    document_words = set(filtered_sentence)     # Remove repeating word set in the document
    features = {}
    for word_set in word_features:   # Loop through all the features selected
        # Enter the feature (Extract the feature from each document) to the dictionary
        features['{} as {}'.format(word_set[0], word_set[1])] = \
            (tuple([word_set[0], word_set[1]]) in document_words)
    return features


def feature_selection(documents):
    word_list = []
    for docs, label in documents:   # Loop through all document
        word_list.extend(docs)  # Append all the word-tag pair
    document_set = set(word_list)   # Remove repeating pair
    all_words = FreqDist((word.lower(), tag) for word, tag in document_set) #Get the distribution
    word_features = list(all_words)[:2000]  # Get the 2000 most common pair
    return word_features
```
Do note that the function is quite similar with only a few differences, due to implementation of POS tagging.
After that we simply proceed as usual, similar steps with basic naive bayes only with the new function.

Resulting in:

![image](https://github.com/ricocahyadi777/pos-tagging-sentiment-analysis-nlp/assets/63791918/f0b5e454-d0fb-4ab2-8433-990cd05455d3)

We can see a decent increase in accuracy. 
Then, we want to try using sentiment polarity calculation as the feature extraction.

## Sentiment polarity calculation

We take a number word with the most extreme sentiment polarity. Polarity range from -1 to 1 (Negative to positive)
Then we can try to run the naïve bayes using the features we extracted. Feature extracted will be the same as the previous one (whether the word tag pair exist in the document). 
But then the word tag pair selected will be the one with the most extreme polarity instead of the most frequent.

To apply sentiment polarity POS tagging for knowledge/lexicon extraction, we need to: 
1. Tag each word with the tag.
2. Get the polarity score.
3. Order it from lowest to highest.

```python
def sentiment_polarity_dictionary_creation(documents):  # Create a sentiment polarity dictionary
    list_of_all_words = []  # Prepare a list to store all the word set (Word with the group tag)
    dictionary_count = {0: {}, 1: {}}   # Create a dictionary for their count based on classification label
    for word_sets, label in documents:   # Loop through the whole data row (Documents)
        non_repeat = set(word_sets)  # Remove repeating word set
        for word_and_tag in non_repeat:    # Loop through the non repeating word set
            list_of_all_words.append(word_and_tag)    # Append the word set into the main list
            if word_and_tag not in dictionary_count[label]: # If the word set is not in the dictionary
                # Make a dictionary for both classification with 0 as the initial count
                dictionary_count[0][word_and_tag] = 0
                dictionary_count[1][word_and_tag] = 0
                dictionary_count[label][word_and_tag] = 1   # Set value as 1 for the label where that word is found
            else:   # If it already existed
                dictionary_count[label][word_and_tag] += 1    # Add 1 to the previous count value
    non_repeating_list = set(list_of_all_words) # Remove repeating word set in the main list
    # Create a polarity score dictionary
    polarity_dictionary = polarity_calculation(dictionary_count, non_repeating_list)
    return polarity_dictionary


def polarity_calculation(dictionary_count, list_of_all_words):  # Create a polarity score dictionary
    # Create the dictionary for all word set with 0 as the initial value
    polarity_dictionary = {word_set: 0 for word_set in list_of_all_words}
    for word, tag in list_of_all_words: # Loop through all the wordset in the documents
        negative_count = dictionary_count[0][(word, tag)]   # Get the count of this word set appear in negative document
        positive_count = dictionary_count[1][(word, tag)]   # Get the count of this word set appear in positive document
        # Get the score for the group tag, 0 for those not in the POS_SCORING
        tag_score = POS_TAG_SCORING[tag] if tag in POS_TAG_SCORING.keys() else 0
        # Calculate the polarity score
        polarity_score = (pow(tag_score, negative_count+1) - pow(tag_score, positive_count+1)) / (1-tag_score)
        polarity_dictionary[(word, tag)] = polarity_score   # Insert the polarity score for the word set
    return polarity_dictionary
```

Then, we get the feature for each document with the help of functions below

```python
def feature_extraction(document, word_features):    # Get the feature for the data row (document)

    stop_words = set(stopwords.words('english'))    # Create the list of stop words

    filtered_sentence = [(w,t) for w, t in document if not w in stop_words] # Filter the sentence by removing stop words
    document_words = set(filtered_sentence)     # Remove repeating word set in the document
    features = {}
    for word_set, value in word_features:   # Loop through all the features selected
        # Enter the feature (Extract the feature from each document) to the dictionary
        features['{} as {}'.format(word_set[0], word_set[1])] = \
            (tuple([word_set[0], word_set[1]]) in document_words)
    return features


def feature_selection(polarity_dictionary, feature_no = 1, positive_n = 750, negative_n = 850):
    # Select features to use
    # Sort the polarity scores in ascending order
    sorted_polarity = sorted(polarity_dictionary.items(), key=lambda kv: (kv[1], kv[0]))
    # Get the n_negative number of most negative polarity
    most_negative_features = sorted_polarity[:negative_n]
    print("Highest in most negative features selected ", most_negative_features[-1])
    # Get the n_positive number of most positive polarity
    most_positive_features = sorted_polarity[-positive_n:]
    print("Lowest in most positive features selected ", most_positive_features[0])

    features1 = []
    features1.extend(most_negative_features)    # Add the negative features into the main list of features
    features1.extend(most_positive_features)    # Add the positive features into the main list of features
    print("Features 1 ({} most positive + {} most negative: {}"
          .format(len(most_positive_features), len(most_negative_features), len(features1)))
    # Feature 2, get all polarity that is not 0 as features
    features2 = [(word_set, score) for word_set, score in sent_dict.items() if score != 0]
    print("Features 2 (Non zero polarity):", len(features2))
    features_selected = []
    if feature_no == 1:  # Select features 1 as the features
        features_selected = features1
        print("Feature 1 is selected")
    elif feature_no == 2:   # Select features 2 as the features
        features_selected = features2
        print("Feature 2 is selected")
    return features_selected
```
As for the features, I prepare 2 features to use. 
* Feature 1 is when you take n-number of most positive polarity and m-number of most negative polarity. 
* Feature 2 is when you get all non-zero polarity.

![image](https://github.com/ricocahyadi777/pos-tagging-sentiment-analysis-nlp/assets/63791918/cc492c40-aa94-4a96-9b83-9afe4fd5478b)

In this code, we are using feature 1, as this provide slightly higher accuracy.
In other dataset, it is possible that feature 2 might give better result.

Feature comparison: <br/>
Feature 1: 
 1.	Parameter (Can be change accordingly, this is the one used as sample): 
  - n most positive = 750 
  -	n most negative = 850
 2.	Accuracy: 85.5%

![image](https://github.com/ricocahyadi777/pos-tagging-sentiment-analysis-nlp/assets/63791918/4f49a418-296c-4678-801b-3907657746e6)

Feature 2:
 1.	Parameter (Can be change accordingly, this is the one used as sample): 
  -	Non-zero polarity
 2.	Accuracy: 85%

![image](https://github.com/ricocahyadi777/pos-tagging-sentiment-analysis-nlp/assets/63791918/b6b4f03d-8b5d-463b-82c9-ae3286b2cd6e)

## Summary
Parameter used:
1.	Out of 1000 data, 800 are used as training, and 200 as test
2.	2000 most frequent words or word-tag pair as features (Naive Bayes Basic & POS-tagging)
3.	Non-zero sentiment polarity as the features (Sentiment Polarity Calculation) 

Final result:
1.	Basic naïve bayes accuracy: 76% 
2.	Naive bayes with pos tag as feature extracted:  81% 
3.	Naive bayes with sentiment polarity as feature extracted: 85.5% 

Note:
We do not do a lot of preprocessing to avoid nltk POS tagging feature to tag wrongly eg. when you stem “Excellent”, it will become excel. Thus, we don’t want wrong tagging. We only ignore stop words during the feature extraction.



