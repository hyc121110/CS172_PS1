# Part A: 
# parse each doc from dir
# for each doc, tokenize doc into words, remove stop words, lowercase each token
import os
import string
import math
from nltk.stem import PorterStemmer
from collections import defaultdict

path = os.getcwd()
ps = PorterStemmer()

word_freq = defaultdict(int) # count for terms in a document
posting_index = dict() # dictionary with key "term" and value (# docs in which term occurs, [posting-list])
document_index = dict() # dictionary with key "doc id" and value "num of terms in doc"

def createIndex(directory):
  directory = "\\" + directory
  with open('stoplist.txt') as f1:
    stop_words = f1.read()
    # define new path
    new_path = path + directory
    # process each files in same directory
    for fn in os.listdir(new_path):
      new_fn = new_path + "\\" + fn
      # open file in directory
      with open(new_fn, "r") as f:
        docID_freq = defaultdict(int) # temporary dictionary for posting-list
        # split words
        words = [word for line in f for word in line.split()]
        # count no. of terms in sentence
        num_terms = 0
        # term frequency count for each file
        word_freq_in_cur_doc = defaultdict(int)
        for word in words:
          # one liner to remove punctuation and lowercase each token            
          word = word.translate(str.maketrans('', '', string.punctuation)).lower()
          # check if word is a stop word
          if word not in stop_words:
            num_terms += 1
            # apply stemming
            word = ps.stem(word)
            docID_freq[word] += 1

            if word in word_freq:
              # word has appeared before, just increment counters
              word_freq[word] += 1
              word_freq_in_cur_doc[word] += 1
            else:
              # initialize values of counters 
              word_freq[word] = 1
              word_freq_in_cur_doc[word] = 1
              posting_index[word] = (1, list())

        # after all words are counted
        document_index[fn] = num_terms
        for term in docID_freq.keys():
          posting_index_list = posting_index[term][1] # a list which contains tuples of doc id and freq of word in doc
          posting_index_list.append((fn, word_freq_in_cur_doc[term])) # append doc id and freq of word in doc
          
  # count the total number of items in posting_index_list
  for term in posting_index.keys():
    doc_cnt = posting_index[term][1]
    # reset the value of "num docs in which term occurs" as number of items in the list
    posting_index[term] = (len(doc_cnt), posting_index[term][1])

def termLookup(query):
  # print tf-idf of query for each doc
  # if term not found, print "No Match"

  # one liner to remove punctuation and lowercase each token            
  word = query.translate(str.maketrans('', '', string.punctuation)).lower()
  # apply stemming
  word = ps.stem(word)

  # check if query in posting_index
  if word in posting_index.keys():
    # calculate tf-idf
    # first calculate term frequency tf in each document
    for doc in document_index.keys():
      freq = 0
      posting_list = posting_index[query][1]
      # find doc id
      for i in range(len(posting_list)):
        if posting_list[i][0] == doc:
          # found doc id, just need the freq
          freq = posting_list[i][1]
          break
      # calculate tf
      tf = freq / document_index[doc]
      # now calculate idf
      N = len(document_index)
      n_k = posting_index[query][0]
      idf = 1 + math.log(N)/(n_k+1)
      # calculate tf-idf
      tfidf = tf * idf
      print("TF-IDF for", doc, "is", tfidf)
  else:
    # no match
    print("No Match")

# main

# prompt user for a directory
while True:
  print("Please enter a directory (dataset1 or dataset2): ", end="")
  directory = input()
  if directory == "dataset1" or directory == "dataset2":
    break
  else:
    print("Please enter a valid directory.")
    continue
createIndex(directory)

# prompt user for a term
print("Please enter a term: ", end="")
query = input()
termLookup(query)