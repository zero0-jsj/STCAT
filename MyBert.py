
import torch.nn as nn
from config import cfg
import copy

# 用预训练Bert构成自己的Bert模型，用于处理文本获取查询特征
class MyBert(nn.Module):

    def __init__(self, corenlp_path, bert_path):
        super().__init__()




    def get_bert_feature(self, sentences):

        # print(sentences)
        # print(sentences)
        if not isinstance(sentences, list):
            sentences = [sentences]

        inputs = self.tokenizer(sentences, max_length=cfg.BERT.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to('cuda:0')
        token_type_ids = inputs['token_type_ids'].to('cuda:0')
        attention_mask = inputs['attention_mask'].to('cuda:0')
        # print(input_ids.shape)
        # print(token_type_ids.shape)

        outputs = self.bert(input_ids, attention_mask, token_type_ids)

        # 查询特征（B,K,D）这里的D为784
        pooler_output = outputs['last_hidden_state']
        # print(pooler_output.shape)

        # 获取名词的mask，大小为（B,K），为1表示该处的特征为名词的特征
        noun_word_index_list = []
        for sentence in sentences:
            sen_tag = self.nlp.pos_tag(sentence)  # 词性标注，其中'NN'表示名词

            noun_word_index = []  # 保存名词
            noun_word_index.extend([0]*token_type_ids.shape[-1])
            for i in range(len(sen_tag)):
                # 判断是否为名词
                if sen_tag[i][1] == 'NN':
                    noun_word_index[i+1] = 1
            noun_word_index_list.append(noun_word_index)

        return pooler_output, noun_word_index_list

    def get_noun_word_feature(self, feature, noun_word_index):
        return feature[noun_word_index, :]