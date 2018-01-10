with open('test.txt') as data:
  word_id_dic={}
  for line in data:
    word_id_list=line.strip().split("\t")
    word=word_id_list[0]
    word_id=word_id_list[1]
    word_id_dic[word]=word_id
  print(len( word_id_dic))
  print(type(word_id_dic['car']))

