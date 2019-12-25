# 基于Adversarial Attack的问题等价性判别

## **背景** 

虽然近年来智能对话系统取得了长足的进展，但是针对专业性较强的问答系统（如法律、政务等），如何准确的判别用户的输入是否为给定问题的语义等价问法仍然是智能问答系统的关键。举例而言，**“市政府管辖哪些部门？”**和**“哪些部门受到市政府的管辖？”**可以认为是语义上*等价*的问题，而**“市政府管辖哪些部门？”**和**“市长管辖哪些部门？”**则为不等价的问题。

 

针对问题等价性判别而言，除去系统的*准确性*外，系统的*鲁棒性*也是很重要、但常常被忽略的一点需求。举例而言，虽然深度神经网络模型在给定的训练集和测试集上常常可以达到满意的准确度，但是对测试集合的稍微改变（Adversarial Attack）就可能导致整体准确度的大幅度下降。



  如以下样例：

| **origin example**             | **adversarial example**        |
| ------------------------------ | ------------------------------ |
| 检察机关提起公益诉讼是什么意思 | 监察机关提起公益诉讼是什么意思 |
| 检察机关提起公益诉讼是什么意思 | 检察机关发起公益诉讼是什么意思 |
| 寻衅滋事一般会怎么处理         | 寻衅兹事一般会怎么处理         |
| 什么是公益诉讼                 | 请问什么是公益诉讼             |



右列数据是左列数据出现一些错别字或者添加一些无意义的干扰词产生的，并不影响原句的意思，但是这样的微小改动可能会导致完全不同结果。从用户的角度而言，这意味着稍有不同的输入就可能得到完全不一样的结果，从而严重降低用户的产品使用体验。



## **数据**

**1. 数据详情**

提供的是一个法律领域的问句等价性数据集，该数据集为我们在实际项目中开发系统所使用的数据集。

训练集根据在实际项目中的数据情况，以问题组的形式提供，每组问句又分为等价部分和不等价部分，等价问句之间互相组合可以生成正样本，等价问句和不等价问句之间互相组合可以生成负样本。我们提供6000组问句的训练集，每组平均有三个等价问句和3个不等价问句。验证集和测试集则以问句对的格式提供，其中验证集有5000条数据。测试集中除了人工标注的样本外，还会有大量adversarial example。

**2. 数据格式**

**train_set.xml**

训练集以XML文件提供，用于训练模型。XML文件中内容格式如下：

```xml
<?xml version="1.0" encoding="utf8"?>

<TrainCorpus>

<Questions number="0">

<EquivalenceQuestions>

<question>什么是公益诉讼？</question>

<question>公益诉讼的定义？</question>

<question>公益诉讼的概念</question>

<question>公益诉讼的定义是什么？</question>

</EquivalenceQuestions>

<NotEquivalenceQuestions>

<question>环境公益诉讼的原告是什么意思？</question>

<question>什么样的鉴定依据算是民事公益诉讼</question>

<question>检察机关提起公益诉讼是什么意思</question>

<question>什么是行政诉讼？</question>

</NotEquivalenceQuestions>

</Questions>

<Questions number="1">

<EquivalenceQuestions>

<question>检察机关提起公益诉讼的目的是什么？</question>

<question>检察机关为什么要提起公益诉讼？</question>

<question>检察机关提起公益诉讼的目的？</question>

</EquivalenceQuestions>

<NotEquivalenceQuestions>

<question>十八大提出由检察机关提起公益诉讼有利于</question>

<question>如何认识检察机关在公益诉讼中的地位</question>

<question>检察机关提起公益诉讼是什么意思</question>

<question>检察机关提起公益诉讼有哪些优势</question>

</NotEquivalenceQuestions>

</Questions>

</TrainCorpus>
```



每一个Questions标签中为一组数据，其中EquivalenceQuestions标签内的问句之间互为等价关系，NotEquivalenceQuestions标签内的问句与EquivalenceQuestions为不等价关系。EquivalenceQuestions之间的问句互相组合可以生成正样本（label为1），EquivalenceQuestions和NotEquivalenceQuestions之间的问句互相组合可以生成负样本（label为0），具体需要生成多少正样本多少负样本由参赛选手自行决定

**dev_set.csv**

此数据集用于测试，数据格式如下：

| **qid** | **question1**                              | **question2**                      |
| ------- | ------------------------------------------ | ---------------------------------- |
| 1       | 醉酒驾驶，保险公司赔偿吗                   | 当事人醉酒驾驶，保险公司会不会赔偿 |
| 2       | 酒驾会吊销驾照吗                           | 醉驾会被吊销驾照吗                 |
| 3       | 被他人的摩托车撞到，导致骨折，对方应赔多少 | 被摩托车撞到骨折，该赔多少         |

 

**test_dev.csv**

此数据集用于最终结果的评定，其格式和dev_set一致，不同的是该数据集中会包含大量adversarial example。

 ![img](https://biendata.com/media/competition/2019/12/09/-1_JjWyL0M.jpg)







## **评测标准**

评测标准为Macro F1值。Macro F1等于每个类别的F1的均值。

对于正样本：

**精确率 = 正确预测标签为1的数量 / 预测标签为1的数量**

**召回率 = 正确预测标签为1的数量 / 真正标签为1的数量**

**F1 = （2 \* 精确率 \* 召回率）/（精确率 + 召回率）**

**负样本是同样的计算方法。**



Macro F1 = (F1_正样本 + F1_负样本) / 2



## **分析**

1. 对抗样本可能是一句话中出现一些错别字或者无意义字词等方式生成的.
2. 添加无意义字词的一般并不会改变原问题的语义.
3. 出现错别字或者不同字的,如果通过语义,能容易判断出真实想说的词或意思的,认为语义没有改变.
4. 最终的测试数据集中包含50%的对抗样本.
5. NotEquivalenceQuestions标签内的问句相互之间可能是等价的,也可能是不等价的.





## 代码结构

- code  代码文件
  - bert4keras/  外部库代码
  - data/  数据处理代码
  - error_correct/  错字生成和纠错代码
  - aa_predict_base_bert.py  训练代码
  - aa_train_base_bert.py  预测代码
  - aa_predict_base_bert.py  预测代码
- data  官方数据、外部词数据和模型源文件
  - bert_roberta/  源模型文件
  - similar_words/  同义词、反义词数据
  - chars.dict  生成的train数据中出现的词（剔除单字的词）
  - law_word.txt  外部收集的法律相关词汇
  - stop_words.txt  外部停用词汇
  - test_set.csv  测试集
  - token_freq.txt  外部结巴词汇表
  - train_set.xml  训练集
- model  存放训练的模型文件
  - bert_res/  训练完成的bert模型文件
  - detect/  训练完成的错词检测模型文件
  - error_maker_save/  纠错模型文件
  - tran_pre/  训练模型文件的词汇表部分
  - tran_pre_for_error_detect/  错词检测模型文件的词汇表部分



## 训练运行

准备Roberta-large模型源文件，解压放到 /data/bert_roberta 文件夹下
其他外部数据文件包括：   

- data  官方数据、外部词数据和模型源文件
  - bert_roberta/  源模型文件
  - similar_words/  同义词、反义词数据
  - law_word.txt  外部收集的法律相关词汇
  - stop_words.txt  外部停用词汇
  - test_set.csv  测试集
  - token_freq.txt  外部结巴词汇表
  - train_set.xml  训练集
    model文件下其它模型文件删除，否则某些模型会继续训练而不是重新开始

```bash
cd code
python3 aa_cfg.py
cd data
python3 aa_data_pre.py
cd ../error_correct
python3 ec_data_pre.py
python3 correct_by_statistics.py
```

此后，训练模型检测模型。（纠错模型在 correct_by_statistics.py 中已执行统计）   
train_error_detect.py 中的 line 75 ~ 76 互相注释，使用加载预训练模型

```bash
python3 train_error_detect.py
```

训练、预测6折交叉验证模型

```bash
cd ..
python3 aa_train_base_bert.py
```

## 预测运行

```
cd code
python3 aa_predict_base_bert.py
```



预测结果保存在 /submit  文件夹下的 result.txt