from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
#turns text data into number data

positive_texts = [
  "we love you",
  "they love us",
  "you are good",
  "he is good",
  "they love mary",
  "i love you",
  "this is not bad"
]

negative_texts = [
  "we hate you",
  "they hate us",
  "you are bad",
  "he is bad",
  "we hate mary",
  "this is not good"
 ]

test_texts = [
  "they love mary",
  "they are good",
  "why do you hate mary",
  "they are almost always good",
  "we are very bad",
  "i hate to love you",
  "this is not bad"
]

training_texts = negative_texts + positive_texts
training_labels = ["negative"] * len(negative_texts) + ["positive"] * len(positive_texts)

# ['we hate you', 'they hate us', 'you are bad', 'he is bad', 'we hate mary', 'we love you', 'they love us', 'you are good', 'he is good', 'they love mary']
# ['negative', 'negative', 'negative', 'negative', 'negative', 'positive', 'positive', 'positive', 'positive', 'positive']

# fitting is turning the words into indices (numbers)
vectorizer = CountVectorizer()
vectorizer.fit(training_texts)

print(vectorizer.vocabulary_)

# {'we': 10, 'hate': 3, 'you': 11, 'they': 8, 'us': 9, 'are': 0, 'bad': 1, 'he': 4, 'is': 5, 'mary': 7, 'love':6, 'good': 2}

# now creating vectors

training_vectors = vectorizer.transform(training_texts)
testing_vectors = vectorizer.transform(test_texts)
# words in the testing set that werent fitted before/that dont appear in training data, will just be ignored in the making of testing vectors

# a decision tree is a classification method

classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)
# compares 'positive'/'negative' and [0120] ALREADY CLASSIFIED vectors to create rules 
predictions = classifier.predict(testing_vectors)
print(predictions)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,5))
tree.plot_tree(classifier,feature_names = vectorizer.get_feature_names_out(), rounded = True, filled = True)
fig.savefig('tree.png')