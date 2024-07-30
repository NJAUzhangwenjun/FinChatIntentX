# -*- coding: utf-8 -*-


# 确保安装所需库
# 安装必要的库
# !pip install transformers torch sklearn pandas torchcrf openpyxl tensorboard

import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# 读取Excel数据
# data = pd.read_excel('path/to/your/data.xlsx')  # 更新为实际路径

# # 数据处理
# texts = data['话术'].tolist()  # 提取'话术'列数据作为文本
# intents = data['意图'].tolist()  # 提取'意图'列数据作为标签

# mock data
texts = [
    "我需要还清所有欠款",
    "请告诉我如何分期还款",
    "客户希望提前还款",
    "如何处理无法按时还款的客户",
    "客户申请延期还款",
    "请确认客户的还款计划",
    "如何联系客户确认还款安排"
]

intents = [
    "全额还款", "分期还款", "提前还款", "无法按时还款", "延期还款", "确认还款计划", "确认还款安排"
]

#----------- 数据预处理-------------
# 编码标签
intent_encoder = LabelEncoder()  # 实例化标签编码器
intent_labels = intent_encoder.fit_transform(intents)  # 将意图标签转换为数值


# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, intent_labels, test_size=0.1, random_state=42
)

# 初始化BERT Tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')  # 使用预训练的BERT Tokenizer
bert_model = BertModel.from_pretrained('yiyanghkust/finbert-pretrain')  # 使用预训练的BERT模型

# 数据转换为BERT输入格式
train_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')  # 对训练文本进行编码
test_inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')  # 对测试文本进行编码

#---------- 定义模型 --------
# 定义意图识别模型
class IntentRecognitionModel(nn.Module):
    """
    意图识别模型，结合BERT、RCNN、双向GRU和全连接层

    此类实现了一个用于金融领域的意图识别模型。它利用预训练的BERT模型生成初始的词嵌入，
    接着通过递归卷积神经网络（RCNN）提取高级特征，然后使用双向门控循环单元（GRU）对序列模型进行处理，
    并以全连接层结束进行分类。

    属性:
        bert (nn.Module): 预训练的BERT模型，用于生成上下文敏感的词嵌入。
        rcnn (nn.Sequential): RCNN模块，用于从BERT嵌入中提取高层次特征。
        gru (nn.GRU): 双向GRU，用于捕捉特征序列中的时间依赖性。
        fc (nn.Linear): 全连接层，用于意图分类。

    参数:
        bert_model (nn.Module): 预训练的BERT模型，用于文本编码。
        num_intents (int): 意图分类的数量。
    """

    def __init__(self, bert_model, num_intents):
        super(IntentRecognitionModel, self).__init__()
        self.bert = bert_model  # BERT模型，用于生成词嵌入

        # RCNN模块，用于特征提取
        self.rcnn = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 双向GRU，用于序列建模
        self.gru = nn.GRU(128, 64, bidirectional=True, batch_first=True)

        # 全连接层，用于意图分类
        self.fc = nn.Linear(64 * 2, num_intents)

    def forward(self, input_ids, attention_mask):
        """
        意图识别模型的前向传播。

        参数:
            input_ids (torch.LongTensor): BERT的输入令牌ID。
            attention_mask (torch.LongTensor): BERT的注意力掩码，标记填充令牌。

        返回:
            torch.FloatTensor: 意图分类的logit值。
        """

        # 使用BERT生成嵌入
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # 应用RCNN进行特征提取
        # 调整维度以适应RCNN输入（batch_size, seq_length, embed_dim）-> （batch_size, embed_dim, seq_length）
        x = self.rcnn(sequence_output.permute(0, 2, 1))

        # 调整维度回（batch_size, seq_length, embed_dim）以适应GRU输入
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)

        # 使用最后一个时间步的隐藏状态进行分类
        x = x[:, -1, :]

        # 通过全连接层进行意图分类
        intent_logits = self.fc(x)

        return intent_logits




# num_intents = len(intent_encoder.classes_)  # 获取意图类别数
# model = IntentRecognitionModel(bert_model, num_intents)  # 实例化意图识别模型

# --------配置超参数---------
learning_rate = 5e-5  # 学习率
batch_size = 2  # 批次大小
num_epochs = 3  # 训练轮数

# --------配置设备和数据------
# 检查设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型迁移到设备

# 数据迁移到设备
train_inputs['input_ids'] = train_inputs['input_ids'].to(device)
train_inputs['attention_mask'] = train_inputs['attention_mask'].to(device)
test_inputs['input_ids'] = test_inputs['input_ids'].to(device)
test_inputs['attention_mask'] = test_inputs['attention_mask'].to(device)
train_labels = torch.tensor(train_labels).to(device)  # 将标签迁移到设备

# -------定义训练和保存函数-------
# 模型保存路径
model_save_path = './intent_recognition_model.pth'

# 先尝试加载模型（如果存在）
if os.path.exists(model_save_path):
    print("加载保存的模型...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))  # 加载保存的模型到指定设备
else:
    print("没有保存的模型，初始化模型...")

# 训练和保存模型
def train_and_save_model():
    # 训练设置
    train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)  # 创建训练数据集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 创建数据加载器（使用可变的批次大小）

    writer = SummaryWriter(log_dir='./logs')  # TensorBoard记录器

    # 将模型图添加到 TensorBoard
    dummy_input = torch.randint(0, 1000, (batch_size, 15))  # 使用虚拟输入数据
    dummy_attention_mask = torch.ones_like(dummy_input)  # 使用虚拟注意力掩码
    writer.add_graph(model, (dummy_input, dummy_attention_mask))  # 添加模型图到 TensorBoard

    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器设置
    loss_fn = nn.CrossEntropyLoss()  # 损失函数

    # 训练过程
    for epoch in range(num_epochs):  # 训练指定轮数
        model.train()  # 设置模型为训练模式
        total_loss = 0  # 初始化总损失
        correct_preds = 0  # 初始化正确预测数
        total_preds = 0  # 初始化总预测数

        for batch in train_loader:  # 遍历训练数据
            optimizer.zero_grad()  # 清零梯度
            input_ids, attention_mask, labels = batch  # 获取批次数据
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)  # 数据迁移到设备
            intent_logits = model(input_ids, attention_mask)  # 获取模型预测
            loss = loss_fn(intent_logits, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()  # 累加损失

            # 计算训练准确率
            _, predicted = torch.max(intent_logits, 1)  # 获取预测标签
            correct_preds += (predicted == labels).sum().item()  # 计算正确预测数
            total_preds += labels.size(0)  # 计算总预测数

        avg_loss = total_loss / len(train_loader)  # 计算平均损失
        train_accuracy = correct_preds / total_preds  # 计算训练准确率
        print(f"Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {train_accuracy}")  # 打印损失和准确率
        writer.add_scalar('Loss/train', avg_loss, epoch)  # 记录损失到TensorBoard
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)  # 记录准确率到TensorBoard

        # 测试模块
        model.eval()
        test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], torch.tensor(test_labels))
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        total_test_loss = 0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                intent_logits = model(input_ids, attention_mask)
                loss = loss_fn(intent_logits, labels)
                total_test_loss += loss.item()
                _, predicted = torch.max(intent_logits, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        avg_test_loss = total_test_loss / len(test_loader)
        accuracy = correct_preds / total_preds
        print(f"Test Loss: {avg_test_loss}, Accuracy: {accuracy}")  # 打印测试结果
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)

    writer.close()  # 关闭TensorBoard记录器

    # 保存模型
    print("保存训练后的模型...")
    torch.save(model.state_dict(), model_save_path)  # 保存模型状态字典

# -----运行训练-----
# 进行训练
train_and_save_model()  # 训练并保存模型

# Commented out IPython magic to ensure Python compatibility.
# 安装 tensorboard
# !pip install tensorboard

# 启动 TensorBoard，指定日志目录
# %load_ext tensorboard
# %tensorboard --logdir=./logs
# %reload_ext tensorboard
