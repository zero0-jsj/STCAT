from MyBert import MyBert
# from stanfordcorenlp import StanfordCoreNLP
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# import torch.nn as nn
# import numpy as np
# import torch
# import nltk
# from nltk.tree import Tree as nltkTree

# corenlp预训练模型地址
corenlp_path = '../corenlp/stanford-corenlp-4.5.1'

# nlp = StanfordCoreNLP(corenlp_path)

# 模拟一个batch的句子
sentence = ["I see you",
            "person removes plate out of cabinet",
            "Season the lettuce with salt and pepper"]    # 输入的句子

# sen_tag = nlp.pos_tag(sentence)   # 词性标注，其中'NN'表示名词
#
# print(sen_tag)
#
# noun_word_index = []    # 保存名词
# for i in range(len(sen_tag)):
#     # 判断是否为名词
#     if sen_tag[i][1] == 'NN':
#         noun_word_index.append(True)
#     else:
#         noun_word_index.append((False))
#
# print(noun_word_index)    # 输出所有名词

# bert预训练模型的地址
bert_path = '../bert/bert-base-uncased'

# 创建bert模型
myBert = MyBert(corenlp_path, bert_path)

# 对句子进行处理，获取查询特征和对应的mask
# [B, K(最大单词数量), D(784)]， [B, K]
# 1 1 0
# 1 0 0
# 1 0 1
sen_feature, noun_word_mask = myBert(sentence)

print(sen_feature.shape)
print(noun_word_mask)

# for i in range(len(sentence)):
#     feature = myBert.get_noun_word_feature(sen_feature[i], noun_word_index_list[i])
#     print(feature.shape)

# tokenizer = BertTokenizer.from_pretrained(vocab_path)
# model = BertModel.from_pretrained(bert_path)
#
# tokenizer_sentence = tokenizer.tokenize(sentence)   # token初始化
# print(tokenizer_sentence)
#
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenizer_sentence)   # 获取词汇表索引
#
# token_tensor = torch.tensor([indexed_tokens])   # 将输入转换为torch的tensor
# print(token_tensor)
#
# with torch.no_grad():    # 禁用梯度计算，因为只是前向传播获取隐藏层状态，所以不需要计算梯度
#     last_hidden_states = model(token_tensor)[0]
#
# # 通过最后四层的连接和求和来创建单词向量
# token_embeddings = []
# for token_i in range(len(tokenizer_sentence)):
#     hidden_layers = []
#     for layer_i in range(len(last_hidden_states)):
#         # 如果输入是单句不分块则中间为0，如果分块还要再遍历一次
#         vec = last_hidden_states[layer_i][0][token_i]
#         hidden_layers.append(vec)
#     token_embeddings.append(hidden_layers)
#
# # 连接最后四层[number_of_tokens, 3072(768*4)]
# concatenated_last_4_layers = torch.Tensor([torch.cat((layer[-1],layer[-2], layer[-3], layer[-4]), 0).cpu().detach().numpy() for layer in token_embeddings])
# print(concatenated_last_4_layers.shape)
# # 对最后四层求和[number_of_tokens, 768]
# summed_last_4_layers = torch.Tensor([torch.sum(torch.stack(layer)[-4:], 0).cpu().detach().numpy() for layer in token_embeddings])
# print(summed_last_4_layers.shape)
#
# summed_last_4_layers = summed_last_4_layers[noun_word_index, :]
# print(summed_last_4_layers.shape)

