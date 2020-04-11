# -*-coding:utf-8 -*-

'''
@File       : hanzi_auto_cut.py
@Author     : HW Shen
@Date       : 2020/3/28
@Desc       : 最大概率汉语切分

要求：
1 采用基于语言模型的最大概率法进行汉语切分。
2 切分算法中的语言模型可以采用n-gram语言模型，要求n >1，并至少采用一种平滑方法；
'''''


import re
import math

MAX_SPLITLEN = 4  # 最大切分长度
corpus_lib = ''  # corpus:语料


def init_corpus_lib(path):  # 初始化语料库
    global corpus_lib
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        corpus_lib = str(file.readlines())


def get_candidate_words(sen):
    """  获取候选词列表 """

    global MAX_SPLITLEN
    global corpus_lib
    candidate_words = []
    for sp in range(len(sen)):
        w = sen[sp]
        candidate_words.append([w, sp, sp])  # 有些字可能不在语料库中，把它作为单个字加进去
        for mp in range(1, MAX_SPLITLEN):  # 判断1 ~ MAX_SPLITLEN-1这3种词中是否有候选词.
            if sp + mp < len(sen):
                w += sen[sp + mp]
                if w in corpus_lib:
                    candidate_words.append([w, sp, sp + mp])  # 存储词，初始位置，结束位置
    print('候选词有：%s' % candidate_words)
    return candidate_words


def segment_sentence(sen):  # sen:sentence 即要切分的句子
    global MAX_SPLITLEN
    global corpus_lib

    candidate_words = get_candidate_words(sen)  # 获取sentence中的所有候选词
    count = 0
    for word in candidate_words:
        if count > 1000:  # 为防止对长句子解析时间过长，放弃一部分精度追求效率
            break
        if word[1] == 0 and word[2] != len(sen) - 1:  # 如果句子中开头的部分，还没有拼凑成整个词序列的话
            no_whitespace_sen = ''.join(word[0].split())
            # word 比如：['今天', 1, 2]，1是 '今' 在句子中的位置，2是 '天' 的位置
            for word in candidate_words:
                if word[1] == 0 and word[2] != len(sen) - 1:
                    end = word[2]
                    for later_word in candidate_words:
                        if later_word[1] == end + 1:  # 如果later_word是当前词的后续词，那么拼接到当前词上
                            word_seq = [word[0] + ' ' + later_word[0], word[1], later_word[2]]  # 合并
                            candidate_words.append(word_seq)
                            print('拼出了新词：%s' % word_seq)
                            count += 1
                    # 遍历完后，这个开头部分短语要移除掉，不然下次遍历还会对它做无用功
                    candidate_words.remove(word)
    print('所有结果词序列有：%s' % candidate_words)

    word_segment_res_list = []  # 存储分词结果序列
    for seque in candidate_words:
        if seque[1] == 0 and seque[2] == len(sen) - 1:
            word_segment_res_list.append(seque[0])
    print('获得的所有分词结果是：')
    print(word_segment_res_list)
    return word_segment_res_list


# P(w1,w2,...,wn) = P(w1|start)P(w2|w1)P(w3|w2).....P(Wn|Wn-1)
# 下标从0开始： = P(w0|start)P(w1|w0)...P(Wn-1|Wn-2)
def calculate_word_sequence_probability(sequence):
    global corpus_lib
    word_list = sequence.split(' ')
    total_word_num = len(corpus_lib)
    prob_total = 0.0
    word_start = word_list[0]
    # 计算第一个词出现的概率P(w1|start)=Count(w1)/total
    count = len(re.findall(r'\s' + word_start + r'\s', corpus_lib)) + 1  # 加1平滑
    prob_total += math.log(count / total_word_num)
    # 计算P(w2/w1)P(w3/w2).....P(Wn/Wn-1)
    for i in range(len(word_list) - 1):  # 0~ n-2
        prev_w = word_list[i]
        later_w = word_list[i + 1]
        count = len(
            re.findall(r'\s' +prev_w +r'\s' +later_w +r'\s',corpus_lib))
        count += 1  # 做一次加1平滑
        prob_total += math.log(count / total_word_num)
    print('%s的概率是：' % sequence)
    print(prob_total)
    return prob_total


def calculate_biggest_prob(word_segm_res):
    best_w_s = ''
    max_prob = 0.0
    for w_s in word_segm_res:  # 改进：先只计算词的数目<=0.6 句子字数的，如果不行再计算全部的概率
        no_whitespace_sen = ''.join(w_s.split())
        zi_shu = len(no_whitespace_sen)
        if len(w_s.split(' ')) <= zi_shu * 0.6:
            prob = calculate_word_sequence_probability(w_s)
            if max_prob == 0 or max_prob < prob:
                best_w_s = w_s
                max_prob = prob
        if best_w_s == '':  # 如果上面的0.6不行的话，再计算全部的概率
            prob = calculate_word_sequence_probability(w_s)
            if max_prob == 0 or max_prob < prob:
                best_w_s = w_s
                max_prob = prob
    print('最好的分词结果（概率为%s）是 ：%s' % (math.pow(math.e, max_prob), best_w_s))
    return best_w_s

# 从中间切分一下，返回中间切分的位置
def split_middle(sen_to_segment):
    length = len(sen_to_segment)
    start = int(length / 2) - 2
    end = start + 5

    # 对中间的5个字进行切分，然后找“第一个空格”，按此把整个句子一分为二
    middle_part = sen_to_segment[start:end]

    best_segm_res = calculate_biggest_prob(segment_sentence(middle_part))
    return start + best_segm_res.index(' ') - 1


# 按任意标点符号划分句子，对每个短句进行分词
def split_mark_and_too_long_sent(sentences):
    sen_list = sentences.splitlines()
    print(sen_list)

    out_text = ''
    for line in sen_list:
        sen_to_segment = ''
        for single_char in line:
            if single_char.isalpha():  # isalpha()表示是否是单词，如果是单词的为True，标点符号等为False
                sen_to_segment += single_char
            elif not single_char.isalpha() and sen_to_segment == '':  # 如果single_char是标点符号、数字,且前面没有待分词的句子
                out_text += single_char + ' '
                print(single_char)

            else:  # 如果single_char是标点符号、数字,
                # 如果句子太长，先从中间切分一下
                if len(sen_to_segment) >= 20:
                    middle = split_middle(sen_to_segment)
                    left_half = sen_to_segment[0:middle + 1]  # 左半部分
                    best_segm_res = calculate_biggest_prob(segment_sentence(left_half))
                    out_text += best_segm_res + ' '
                    # 右半部分交给后面几行处理
                    sen_to_segment = sen_to_segment[middle + 1:len(sen_to_segment)]

                best_segm_res = calculate_biggest_prob(segment_sentence(sen_to_segment))
                print(single_char)
                sen_to_segment = ''
                out_text += best_segm_res + ' ' + single_char + ' '  # 标点两侧也用空格隔起来

        # 如果这行句子最后还有一些文字没有切分的话
        if sen_to_segment != '':
            best_segm_res = calculate_biggest_prob(segment_sentence(sen_to_segment))
            out_text += best_segm_res + ' '
        out_text += '\n'

    with open('D:/1佩王的文件/计算语言学基础/生成结果.txt', 'w') as file:
        file.write(out_text)
    print(out_text)


if __name__ == '__main__':
    path = 'D:/1佩王的文件/计算语言学基础/北大(人民日报)语料库199801.txt'
    init_corpus_lib(path)  # 初始化语料库

    sentences = ''
    path = 'E:/study/1.研一的课/计算语言学基础课件/testset.txt'  # 读取要切分的文章
    with open(path, 'r', encoding='gbk', errors='ignore') as file:
        for line in file.readlines():
            sentences += line

    # 改进：先对句子按标点符号划分成多个短句，然后对每个短句进行切分、计算概率
    split_mark_and_too_long_sent(sentences)


"""
实现思路

1、处理语料库
用的是人民日报语料库，然后为了方便把属性去掉了，只留下了词。

2、读要分词的文本，按照标点符号、数字进行分割
按标点符号、数字进行分割，确保分割结果是只有汉字的句子。如果句子过长(>=20)，则先对句子中间位置的5个字先切分一次，
从5个字的切分结果的第一个空格处，把句子分成两部分，再对每一部分分别切词。标点符号、数字则按照原样输出。

3、找出所有候选词
从一个句子中找出所有的候选词。如每次取4个字，假设为abcd这四个字，得到：a\b\c\d\ab\bc\cd\abc\bcd\abcd，
判断它们每个是否在语料库中，如果是的话则存为候选词。并存储下这个词在句子中的开始位置和结束位置。

4、计算出一个句子所有的切分结果
所有的候选词放到了一个python的list（即集合）中，遍历所有开始位置为0但结结束位不为0的候选词，
按照词的开始位置和结束位置进行拼凑，新拼凑出的元素会加入到这个list中。
当一个词和其他所有能拼凑的词拼凑完后，从list中删除这个词。当遍历结束后，集合中会有长度等于句子长度的元素，
这些元素就是一个句子所有的切分结果。

5、使用2-gram模型计算出每种切分结果的概率，挑选出最大概率的句子切分结果
计算概率时使用条件概率，使用加一平滑。条件概率的公式为：P(w1,w2,…,wn) = P(w1|start)P(w2|w1)P(w3|w2)…..P(Wn|Wn-1)，
利用log把乘法变成加法：log P(w1,w2,…,wn) = log P(w1/start) + logP(w2/w1) + ….. + logP(Wn/Wn-1)
句子往往不是由很多个单字组成的，所以为了提高速度，我们先计算出切分后词个数 <= 0.6*句子字数的切分结果的概率，
如果不为0则返回这个最大概率，如果为0的话，再计算 >= 0.6 的切分结果中的最大概率。

6、将拥有最大概率的句子切分结果存到文件中

"""

