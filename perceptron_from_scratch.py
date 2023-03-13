import re
import random
import pyconll
import json
import matplotlib.pyplot as plt
from collections import defaultdict


class Perceptron:

  def __init__(self, labels):
    self.labels = labels
    self.weights = defaultdict(dict)
    self.features = list()

    #self.param_structure()


  def feature_vector(self, sent, word_ind):
    feat_vec = []
    feat_vec.append("bias")

    # current word   
    feat_vec.append(f"curr_word_{sent[word_ind]}") 

    # previous word    
    if word_ind != 0:
      feat_vec.append(f"prev_word_{sent[word_ind - 1]}")   
    else:
      feat_vec.append("prev_word_<s>")
      feat_vec.append("prev_prev_word_</s>")
  
    # word i - 2
    if word_ind - 1 > 0:
      feat_vec.append(f"prev_prev_word_{sent[word_ind - 2]}")
    elif word_ind == 1:
      feat_vec.append("prev_prev_word_<s>")

    # following word
    if word_ind != (len(sent) - 1):    
      feat_vec.append(f"next_word_{sent[word_ind + 1]}") 
    else:
      feat_vec.append("next_word_</s>")
      feat_vec.append("next_next_word_<s>")

    # word i + 2       
    if word_ind < (len(sent) - 2):
      feat_vec.append(f"next_next_word_{sent[word_ind + 2]}")  
    elif word_ind == len(sent) - 2:
      feat_vec.append("next_next_word_</s>")


    # if starts with capital    
    if sent[word_ind].istitle():
      feat_vec.append("starts_with_upper")
    else:
      feat_vec.append("starts_with_lower")   

    # if contains number:  
    num_val = ""

    if (re.search(r'\d', sent[word_ind])):
      num_val = "contains_number"

    else:
      num_val = "contains_no_numbers"
    feat_vec.append(num_val)


    return feat_vec

  def get_all_feats(self, corpus):
    vector_of_features = []

    for sentence in corpus:
      for word_ind, word in enumerate(sentence["observation"]):
      
        non_zero_feats = self.feature_vector(sentence["observation"], word_ind)
        vector_of_features.extend(non_zero_feats)

    self.features.extend(list(set(vector_of_features)))
    #print(vector_of_features)
    return None

  def param_structure(self):
    for label in self.labels:
      for feat in self.features:
        if feat == "bias":
          self.weights[label][feat] = 1 #0
        else:
          #self.weights[label][feat] = random.uniform(-1e-4, 1e-4)
          self.weights[label][feat] = 0
  
    return None


  def dot(self, observation, param_vec):
    res = 0
    for feat in observation:
      if feat not in param_vec:
        continue
      res += param_vec[feat]
    return res


  def predict(self, observation):
    scores = defaultdict(float)

    for clas, param_vec in self.weights.items():
      scores[clas] = self.dot(observation, param_vec)

    #print(scores)

    return max(scores, key=scores.get)

  def score(self, corpus_vectorized):
    pred_labels = []
    gold_labels = []

    for example in corpus_vectorized:
      gold_labels.append(example[1])
      pred_labels.append(self.predict(example[0])) 
      
    num_correct = 0
    for gold, pred in zip(gold_labels, pred_labels):
      if gold == pred:
        num_correct += 1

    return round(num_correct/len(gold_labels), 3)


  def update_single_example(self, observation, label):
    pred_label = self.predict(observation)

    if pred_label != label:
      for feat in observation:
        self.weights[label][feat] += 1
        self.weights[pred_label][feat] -= 1

    return None

  
  def fit(self, corpus_train_vectorized, corpus_test_vectorized, max_iter):
    score_log_train = []
    score_log_test = []

    for i in range(max_iter):
      # shuffle training data
      random.shuffle(corpus_train_vectorized)

      for observ, label in corpus_train_vectorized:
        self.update_single_example(observ, label)

      score_log_train.append(self.score(corpus_train_vectorized))
      score_log_test.append(self.score(corpus_test_vectorized))

    return score_log_train, score_log_test
  


def file_to_examples(filename):

  gsd_json = []

  for sentence in pyconll.load_from_file(filename):
    tokenized_sentence = []
    gold_labels = []

    for token in sentence:
      # remove multiword tokens
      if "-" not in token.id: 
        tokenized_sentence.append(token.form)
        gold_labels.append(token.upos)

    gsd_json.append({"observation": tokenized_sentence, "label": gold_labels})

  with open('examples.json', 'w') as f:
    json.dump(gsd_json, f)

  #files.download('examples.json')

  return json.load(open("examples.json", "r"))

# function that iterates over a list of dict(observation, label)
# creates new dict(feature_vec, label)
# for each word in observation, apply feature_vec function and 
# add the vec to feature_vec
# add corresponding label to label
def feature_vector(sent, word_ind):

  feat_vec = []
  
  feat_vec.append("bias")

  # current word   
  feat_vec.append(f"curr_word_{sent[word_ind]}") 

  # previous word    
  if word_ind != 0:
    feat_vec.append(f"prev_word_{sent[word_ind - 1]}")   
  else:
    feat_vec.append("prev_word_<s>")
    feat_vec.append("prev_prev_word_</s>")
 
  # word i - 2
  if word_ind - 1 > 0:
    feat_vec.append(f"prev_prev_word_{sent[word_ind - 2]}")
  elif word_ind == 1:
    feat_vec.append("prev_prev_word_<s>")

  # following word
  if word_ind != (len(sent) - 1):    
    feat_vec.append(f"next_word_{sent[word_ind + 1]}") 
  else:
    feat_vec.append("next_word_</s>")
    feat_vec.append("next_next_word_<s>")

  # word i + 2       
  if word_ind < (len(sent) - 2):
    feat_vec.append(f"next_next_word_{sent[word_ind + 2]}")  
  elif word_ind == len(sent) - 2:
    feat_vec.append("next_next_word_</s>")

      
  # if starts with capital    
  if sent[word_ind].istitle():
    feat_vec.append("starts_with_upper")
  else:
    feat_vec.append("starts_with_lower")   

  # if contains number:  
  num_val = ""
  
  if (re.search(r'\d', sent[word_ind])):
    num_val = "contains_number"
      
  else:
    num_val = "contains_no_numbers"
  feat_vec.append(num_val)

  
  return feat_vec


def vectorize_corpus(corpus, vectorize=feature_vector):
  vectorized_corpus = []

  for sentence in corpus:
    for word_ind, word in enumerate(sentence["observation"]):
      vectorized_corpus.append((vectorize(sentence["observation"], word_ind), sentence["label"][word_ind]))

  return vectorized_corpus


def all_tags(corpus):
  tags = set()
  for example in corpus:
    for label in example["label"]:
      tags.add(label)

  return list(tags)


corpus_train = file_to_examples('fr_gsd-ud-train.conllu')
corpus_test = file_to_examples('fr_gsd-ud-test.conllu')

corpus_train_vectorized = vectorize_corpus(corpus_train)
corpus_test_vectorized = vectorize_corpus(corpus_test)

labels = all_tags(corpus_train)

percep = Perceptron(labels)
percep.get_all_feats(corpus_train)
percep.param_structure()

accuracy_scores = (percep.fit(corpus_train_vectorized, corpus_test_vectorized, 10))



print(accuracy_scores)

y_axis = accuracy_scores[1]
#print(y_axis)
x_axis = [x for x in range(len(y_axis))]

plt.plot(x_axis, y_axis)
plt.title("Perceptron accuracy vs # of epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
