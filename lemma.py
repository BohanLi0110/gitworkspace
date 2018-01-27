from stanfordcorenlp import StanfordCoreNLP
import re
import sys


def read_data(path):
  lines=[]
  with open (path) as f:
    for line in f:
      lines.append(line.strip())
  return lines

def tokenize(sent):
  word_list=[x.strip().lower() for x in re.split('(\W)', sent) if x.strip()]
  line=""
  for w in word_list:
    line+=w
    line+=" "
  return line

def split_ph(data):
  p_list=[]
  h_list=[]
  for line in data:
    line_list=line.split("\t")
    p=tokenize(line_list[0])
    h=tokenize(line_list[1])
    p_list.append(p)
    h_list.append(h)
  return p_list,h_list

def core_nlp_lemma(data):
  data_list=[]
  nlp = StanfordCoreNLP(r'../../gpu1/StanfordNLP/stanford-corenlp-full-2017-06-09/', memory='8g')
  props = {'annotators': 'lemma','pipelineLanguage':'en','outputFormat':'text'}
  for line in data:
    lemma_data=nlp.annotate(line, properties=props)
    word_list=get_lemma_word(lemma_data,line)
    data_list.append(word_list)
  return data_list

def get_lemma_word(lemma_data,o_line):
  word_list=[]
  line = lemma_data.split('\n')
  for i in line:
    matchObj = re.match(r'.*Lemma=(.*)]',i)
    if (matchObj):
      #print(id)
      current_word=matchObj.group(1)
      word_list.append(current_word)
  return word_list 
    
def write_data(data_p,data_h,path):
  print(data_p)
  '''data is a 1D list or a set'''
  with open(path,"w") as f:
    for i in range(len(data_p)):
      p=data_p[i]
      h=data_h[i]
      for i in range(len(p)-1):
        f.write(p[i]+" ")
      f.write(p[len(p)-1]+"\t")
      for j in range(len(h)-1):
        f.write(h[j]+" ")
      f.write(h[len(h)-1]+"\n")
      
      """
      for word in p:
        f.write(word+" ")
      f.write("\t")
      for word in h:
        f.write(word+" ")
      f.write("\n")
      """
   
       
def main():
  raw_data=read_data('../data/snli_1.0/train.txt')
  p_list,h_list=split_ph(raw_data)
  lemma_p_list=core_nlp_lemma(p_list)
  lemma_h_list=core_nlp_lemma(h_list)
  write_data(lemma_p_list,lemma_h_list,'lemma_train.txt')
 

if __name__ =='__main__':
  main()  

     
