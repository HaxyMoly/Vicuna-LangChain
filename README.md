# Vicuna-LangChain
A simple LangChain-like implementation based on Sentence Embedding+local knowledge base, with Vicuna (FastChat) serving as the LLM. Supports both Chinese and English, and can process PDF, HTML, and DOCX formats of documents as knowledge base.

一个简单的类LangChain实现，基于Sentence Embedding+本地知识库，以Vicuna作为生成模型。支持中英双语，支持pdf、html和docx格式的文档作为知识库。[中文文档](#简介)


## Introduction
This is a very simple LangChain-like implementation. The detailed implementation is as follows: 
1. Extract the text from the documents in the knowledge base folder and divide them into text chunks with sizes of `chunk_length`.
2. Obtain the embedding of each text chunk through the [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese) model.
3. Calculate the cosine similarity between the embedding of the question and the embedding of each text chunk.
4. Return the top `k` text chunks with the highest cosine similarity and generate a prompt with those.
5. Replace the prompt history with the initial question
6. Generate response with Vicuna according to the prompt.


## Demo
> Tell me about the Wei Lun Hall at The University of Hong Kong (A student dormitory)

❌ Without knowledge base -> Fabricated response
![Without knowledge base](https://github.com/HaxyMoly/Vicuna-LangChain/raw/main/img/without_kb_en.png)
✅ With knowledge base -> Factual responce
![With knowledge base](https://github.com/HaxyMoly/Vicuna-LangChain/raw/main/img/with_kb_en.png)

## Install
1. Clone FastChat repository
```bash
git clone https://github.com/lm-sys/FastChat.git
```
2. Add the following content to line 86 of FastChat/fastchat/conversation.py
```python3
def correct_message(self, message):
        self.messages[-2][-1] = message
```
3. Switch to FastChat/, and install the modified FastChat
```bash
cd FastChat
pip install .
```
4. Obtain the weight of Vicuna v1.1 follow the [instruction](https://github.com/lm-sys/FastChat#vicuna-weights)
5. Clone this repository and install the requirements
```bash
git clone https://github.com/HaxyMoly/Vicuna-LangChain.git
pip install -r requirements.txt
```
6. Create a folder named `documents`, and put your documents in it. Note that only PDF, HTML, and DOCX documents are supported.
```bash
cd Vicuna-LangChain
mkdir documents
```
7. Have fun!
```bash
# With knowledge base
python vicuna_cli.py --vicuna-dir /path/to/vicuna/weights --knowledge-base

# Without knowledge base
python vicuna_cli.py --vicuna-dir /path/to/vicuna/weights --knowledge-base
```

## 简介
本项目是一个简单的langchain-like实现，具体实现如下：
1. 提取知识库文件夹中的文档文本，分割成`chunk_length`大小的文本块
2. 通过[shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)模型计算各文本块的嵌入
3. 计算问题文本嵌入和各文本块的嵌入的余弦相似度
4. 返回余弦相似度最高的`k`个文本作为给定信息生成prompt
5. 将prompt历史替换为最初问的问题
6. 将prompt交给vicuna模型生成答案


## 演示
> 你知道病毒和基因编辑有什么关系吗？

❌ 不带知识库：
![没有知识库](https://github.com/HaxyMoly/Vicuna-LangChain/raw/main/img/without_kb_zh.png)
✅ 带知识库：
![有知识库](https://github.com/HaxyMoly/Vicuna-LangChain/raw/main/img/with_kb_zh.png)

## 安装
1. 克隆FastChat仓库
```bash
git clone https://github.com/lm-sys/FastChat.git
```
2. 在FastChat/fastchat/conversation.py的第86行添加以下内容
```python3
def correct_message(self, message):
        self.messages[-2][-1] = message
```
3. 进入FastChat/目录，安装修改后的FastChat
```bash
cd FastChat
pip install .
```
4. 按照 [Vicuna说明](https://github.com/lm-sys/FastChat#vicuna-weights)获得Vicuna v1.1的权重
5. 克隆本仓库并安装依赖
```bash
git clone https://github.com/HaxyMoly/Vicuna-LangChain.git
pip install -r requirements.txt
```
6. 创建一个名为`documents`的文件夹，将你要作为知识库的文档放进去。注意，仅支持PDF、HTML和DOCX格式的文档。
```bash
cd Vicuna-LangChain
mkdir documents
```
7. 试试看吧
```bash
# 启用知识库
python vicuna_cli.py --vicuna-dir /path/to/vicuna/weights --knowledge-base

# 不启用知识库
python vicuna_cli.py --vicuna-dir /path/to/vicuna/weights --knowledge-base
```