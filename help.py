import re

def create_dic(path):
  word_id_dic = {}
  with open(path) as f:
    for line in f:
      l = line.split('\t')
      word = l[0]
      w_id = l[1]
      word_id_dic[word] = w_id
  return word_id_dic

def read_data(path):
  file_list=[]
  with open (path) as f:
    for line in f:
      file_list.append(line.strip())
    return file_list

def split_ph(data):
  p_list = []
  h_list = []
  for line in data:
    ph_list = line.split("\t")
    p_list.append(tokenize(ph_list[0]))
    h_list.append(tokenize(ph_list[1]))
  return p_list,h_list

def tokenize(sent):
  word_list=[x.strip().lower() for x in re.split(' ', sent) if x.strip()]
  return word_list

def get_id(word,word_id_dic):
  return word_id_dic[word]

def lable(p_id,h_id,triple_list):
  for line in triple_list:
    temp_list = re.split('\t',line)
    #print(temp_list[0]+p_id+temp_list[1]+h_id)
    if int(temp_list[0]) == int(p_id) and int(temp_list[1]) == int(h_id):#int only
      
      if temp_list[2] == 'Synonym':
        return '0'
      elif temp_list[2] == 'Antonym':
        return '1'
      break
  else:
    #print(line+'\t'+'no'+str(i))
    return '3'

def main():
  word_id_dic = create_dic('../data/lemma_vocab.txt')
  triple_list = read_data('../data/triple_100.txt')
  data = read_data("../data/train_100.txt")
  p_list,h_list = split_ph(data)
  
  '''
  Synonym 0
  Antonym 1
  Same 2
  NoRel 3
  '''
  with open('../data/word_rel.txt','w') as result: 
    for i in range(len(p_list)):
      p=p_list[i]
      h=h_list[i]
      for p_word in p:
        p_id = get_id(p_word,word_id_dic)
        for h_word in h:
          h_id = get_id(h_word,word_id_dic)
          if p_id == h_id:
            result.write('2'+' ')
            #print(type(p_id))
          else:
            result.write(lable(p_id,h_id,triple_list)+' ')
      result.write('\n')

if __name__ == '__main__':
  main()

