# paper_generation
本项目实现的功能是从题目中选择题目组合成试卷，实现的两个部分分别为相似试卷的生成和自由组成一张试卷。
以下是使用的具体流程。
本项目需要的库放在requirements.txt 文件夹下，可以直接下载。
首先你需要一个题库，在这个题库基础上使用我的代码可以实现上述功能。
有了一个题库之后，按照每一行是题目编号、题目类型编号、题目类型的汉语、题目难度、题目难度编号、题目知识点编号、题目知识点、题目具体题干内容、题目答案，按照这样的顺序存储题目并保存。
首先生成embedding，使用m3e-timu.py，可以在embedding目录下生成文件。
运行chroma-build-timu.py文件可以建立需要的向量数据库。

## 相似试卷生成
对于相似试卷的生成，首先保存你已有的试卷，去掉无关信息，只留下题目本身。
运行opt.py可以直接使用向量数据库寻找到最相似的题目。

## 自由组合试卷
运行rl_genarated.py 使用这个文件可以使用PPO对模型进行训练，训练的模型保存在ckpt文件夹中。
运行generated.py可以进行试卷生成任务，并且生成新试卷的难度、知识点和题型分布图。
