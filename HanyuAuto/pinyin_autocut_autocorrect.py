# -*-coding:utf-8 -*-

'''
@File       : pinyin_autocut_autocorrect.py
@Author     : HW Shen
@Date       : 2020/3/28
@Desc       : 汉语拼音的 "自动分割" auto-cut 及 "自动纠错" auto-correction
'''''


import re
import numpy as np
import pinyin
from collections import Counter, defaultdict
from functools import lru_cache  # 设置存储限制
from functools import wraps  # 装饰器包
import time


class PinyinAutoCut:

    _MAX_SPLITLEN = 6  # pinyin组合的最大切分长度

    def __init__(self, sentence=''):

        self.PINYIN_CHARATERS = []  # 单个pinyin列表
        self.PINYIN_CHARACTERS_2GRAM = []  # 2_gram pinyin列表

        self.PINYIN_COUNT = {}  # 单个pinyin的词频表
        self.PINYIN_COUNT_2GRAM = {}  # 2_gram组合的pinyin词频表

        self.candidates_pinyin = []  # pinyin的候选组合
        self.pinyin_sequence_combines = []  # 成功拼接的pinyin组合序列

        self.solutions = []  # 每个pinyin序列的概率(pinyin_seq, probability)

    def chinese2pinyin(self, corpus):
        """ 中文转pinyin"""
        # pinyin.get('你好，中国',format='strip',delimiter=' ') => 'ni hao ， zhong guo'
        return pinyin.get(corpus, format='strip', delimiter=' ')

    def token(self, text):
        """ 正则去掉非pinyin的数字和符号等 """
        return re.findall('[a-z]+', text.lower())

    def init_corpus_libary(self, path):
        """读入语料库，构建pinyin库"""

        chinese_dataset = path
        chinese_corpus = open(chinese_dataset).read()

        # 单个pinyin
        self.PINYIN_CHARATERS = self.chinese2pinyin(chinese_corpus)
        self.PINYIN_COUNT = Counter(self.token(self.PINYIN_CHARATERS)) # pinyin 出现的频率

        # 2-gram pinyin
        self.PINYIN_CHARATERS_2GRAM = [''.join(self.PINYIN_CHARATERS[i:i + 2]) for i in range(len(self.PINYIN_CHARATERS[:-2]))]
        self.PINYIN_COUNT_2GRAM = Counter(self.PINYIN_CHARATERS_2GRAM) # 2_gram的pinyin出现频率

    def get_candidates_pinyin(self, sen):
        """  获取候选pinyin列表 """

        for sp in range(len(sen)):
            py = sen[sp]
            self.candidates_pinyin.append([py, sp, sp])  # 有些音节可能不在pinyin库中，把它作为单个音节a,e,o加进去

            # 这里查找的是长度在1-6之间的所有
            for mp in range(1, self._MAX_SPLITLEN+1):  # 判断1-MAX_SPLITLEN这6种中是否有候选词.
                if sp + mp < len(sen):
                    py += sen[sp + mp]
                    if py in self.PINYIN_COUNT:
                        self.candidates_pinyin.append([py, sp, sp + mp])  # 存储词，初始位置，结束位置

    @lru_cache(maxsize=2 ** 10)  # maxsize 设置存储上限，超过上限就把低频的去掉
    def get_pinyin_senquence(self, sen):
        """ 找出pinyin的所有可能的拆分组合 """

        # 获取sentence中的所有候选词
        self.get_candidates_pinyin(sen)
        # pinyin 比如：['yi', 1, 2]，1是 'yi' 在句子中y的位置，2是i的位置
        for pinyin in self.candidates_pinyin:
            if pinyin[1] == 0 and pinyin[2] != len(sen) - 1:
                end = pinyin[2]
                for later_pinyin in self.candidates_pinyin:
                    if later_pinyin[1] == end + 1:  # 如果later_word是当前词的后续词，那么拼接到当前词上
                        new_candidate = [pinyin[0] + ' ' + later_pinyin[0], pinyin[1], later_pinyin[2]]  # 合并
                        self.candidates_pinyin.append(new_candidate)
                        print('拼出了新词：%s' % new_candidate)
                # 遍历完后，这个开头部分短语要移除掉，不然下次遍历还会对它做无用功
                # candidates_pinyin.remove(word)
        print('所有结果词序列有：\n', self.candidates_pinyin)
        for pinyin in self.candidates_pinyin:
            if pinyin[1] == 0 and pinyin[2] == len(sen)-1:
                self.pinyin_sequence_combines.append(pinyin[0])
        print('获得的所有pinyin组合结果：', self.pinyin_sequence_combines)

    def calculate_prob_1(self, py):
        """ 通过+1平滑处理单个pinyin的prob """
        if py in self.PINYIN_COUNT:
            return self.PINYIN_COUNT[py] / len(self.PINYIN_CHARATERS)
        else:
            return 1 / len(self.PINYIN_CHARATERS)

    def calculate_prob_2(self, py1, py2):
        """ 通过2_grams和+1平滑计算2个拼音的prob """
        if py1 + py2 in self.PINYIN_COUNT_2GRAM:
            return self.PINYIN_COUNT_2GRAM[py1 + py2] / len(self.PINYIN_CHARACTERS_2GRAM)
        else:
            return 1 / len(self.PINYIN_CHARACTERS_2GRAM)

    def calculate_pinyin_sequence_probability(self):
        """
        通过n_grams语言模型计算出的2_grams的pinyin候选组合概率:
            P(w1,w2,w3...wn)=P(w1|w2)P(w2|w3)P(w3|w4)P(w4|w5)...P(wn)
        取log乘法变加法 =>
            logP(w1,w2,w3...wn)=logP(w1|w2)+logP(w2|w3)+logP(w3|w4)+logP(w4|w5)+...+logP(wn)
        """
        for combine in self.pinyin_sequence_combines:
            pinyins = combine.split(' ')
            prob = 0.0
            for i in range(len(pinyins) - 1):
                pre_py = pinyins[i]
                later_py = pinyins[i + 1]
                # P(w_i|w_i+1) = P(w_iw_i+1)/P(w_i+1)
                prob += np.log(self.calculate_prob_2(pre_py, later_py) / self.calculate_prob_1(later_py))

            prob += self.calculate_prob_1(pinyins[-1])
            self.solutions.append((combine, prob))

        # 返回prob最大的pinyin序列结果
        return max(self.solutions, key=lambda x: x[0])

    def split_middle(self, sentence):
        """ 从中间切分一下，返回中间切分为的位置 """
        middle = int(len(sentence) / 2)
        # 对中间的8个音节进行切分，然后找“第一个空格”，按此把整个句子一分为二
        start, end = middle - 4, middle + 4
        middle_part = sentence[start: end]
        # print("middle_part: ", middle_part)
        candidates_pinyin_middle = self.get_candidates_pinyin_middle(middle_part)
        # print("candidates: ", candidates)
        pinyin_combines = self.sentence_combine(middle_part, candidates)
        # print("pinyin_combines: ", pinyin_combines)
        optimized_combine, prob = self.calculate_pinyin_split_probability(pinyin_combines)
        # print("optimized_combine: ", optimized_combine)
        return start + optimized_combine.index(' ')


# 计时装饰器
def get_time(func):
    def wrapper(*args):
        start_time = time.time()
        func(*args)
        end_time = time.time()
        print('used_time: {}'.format(end_time-start_time))
    return wrapper


# 存储器装饰器
def memo(f):
    memo.already_computed = {}
    @wraps(f)
    def _wrap(arg):
        if arg in memo.already_computed:
            # 将已经计算过的结果存入memo，下次如果再遇到相同的arg，就直接返回memo中对应的结果
            result = memo.already_computed[arg]
        else:
            # 如果没有结果，就通过 f() 去求解，并将求解的结果存入memo
            result = f(arg)
            memo.already_computed[arg] = result
        return result
    return _wrap


if __name__ == '__main__':

    path = 'article_9k.txt'

    pass




