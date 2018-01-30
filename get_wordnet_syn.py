from nltk.corpus import wordnet as wn
import re

data='data/snli_1.0/test_10.jsonl'

def get_parse(data):
  sen1=[]
  sen2=[]
  with open(data) as f:
    for line in f:
      matchObj1 = re.search(r'sentence1_parse\": \"(.*?)\"',line)
      sen1.append(matchObj1.group(1))
      matchObj2 = re.search(r'sentence2_parse\": \"(.*?)\"',line)
      sen2.append(matchObj2.group(1))
  #print(sen2)
  return sen1,sen2

def parsing_parse(sen1,sen2):
  whole_pair1=[]
  whole_pair2=[]
  for line in sen1:
    base_parse1 = [s.rstrip(" ").rstrip(")") for s in line.split("(") if ")" in s]
    parse_list1 = [pair.split(" ") for pair in base_parse1]
    whole_pair1.append(parse_list1)
  #print(whole_pair1)
  #print('\n')
  for line in sen2:
    base_parse2 = [s.rstrip(" ").rstrip(")") for s in line.split("(") if ")" in s]
    parse_list2 = [pair.split(" ") for pair in base_parse2]
    whole_pair2.append(parse_list2)
    #pos = [pair.split(" ")[0] for pair in base_parse]
  #print(whole_pair2)
  return whole_pair1,whole_pair2

def get_normal_pair(whole_pair1,whole_pair2):
  '''
  n,v,a,r only
  word.lower()
  '''
  pair1=[]
  pair2=[] 
  for line in whole_pair1:
    line_list1=[word for word in line if re.match('NN|VB|JJ|RB',word[0])]
    for word in line_list1:
      if re.match('NN',word[0]):
        word[0]='n'
        word[1]=word[1].lower()
      elif re.match('VB',word[0]):
        word[0]='v'
        word[1]=word[1].lower()
      elif re.match('JJ',word[0]):
        word[0]='a'
        word[1]=word[1].lower()
      elif re.match('RB',word[0]):
        word[0]='r'
        word[1]=word[1].lower()
    pair1.append(line_list1)

  for line in whole_pair2:
    line_list2=[word for word in line if re.match('NN|VB|JJ|RB',word[0])]
    for word in line_list2:
      if re.match('NN',word[0]):
        word[0]='n'
        word[1]=word[1].lower()
      elif re.match('VB',word[0]):
        word[0]='v'
        word[1]=word[1].lower()
      elif re.match('JJ',word[0]):
        word[0]='a'
        word[1]=word[1].lower()
      elif re.match('RB',word[0]):
        word[0]='r'
        word[1]=word[1].lower()
    pair2.append(line_list2)
  return pair1,pair2

def get_synsets(pair1):
  synset_list=[] #whole synset list for all p sentences.
  #str='%s.'%word+'%s.'%pos+'01'
  for line in pair1:
    line_list=[] #synset list for one p sentence.
    for word in line:
      word_dic={}
      word_list=[] #synset list for one word in p.
      pos=word[0]
      w=word[1]
      if len(wn.synsets(w))==0:
        word_list.append('wordnet NOT FOUND '+w+'!')

      else:
        for synset in wn.synsets(w):
          synset_str=str(synset)

          if pos=='a' and re.search(r'Synset\(\'.*?.a|s.[0-9]',synset_str):
            for syn_word in synset.lemma_names():
              if syn_word not in word_list:
                word_list.append(syn_word)
          elif re.search(r'Synset\(\'.*?.%s.[0-9]'%pos,synset_str): #match the specific pos
            for syn_word in synset.lemma_names():
              if syn_word not in word_list:
                word_list.append(syn_word)
      line_list.append(word_list)        
    synset_list.append(line_list)           
  return synset_list
    

'''
def test():
  word = 'right'
  #word1 = 'field'
  synset_list=wn.synsets(word)

  for single in synset_list:
    #Synset('right.r.07')
    matchObj = re.match( r'Synset\(\'%s\.'%word, str(single))
    #matchObj = re.search( r'%s?'%word, str(single))

    if matchObj:
      offset = str(single.offset()).zfill(8)
      print(offset)
      #print(str(single)+'##'+offset)
      #print(type(offset))
'''
def main():
  sen1,sen2 = get_parse(data)    
  whole_pair1,whole_pair2 = parsing_parse(sen1,sen2)
  pair1,pair2 = get_normal_pair(whole_pair1,whole_pair2)
  synset_list = get_synsets(pair1)

if __name__ == '__main__':
  main()
