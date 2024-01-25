# CVPR'21论文代码: "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# 作者: Can Zhang*, Meng Cao, Dongming Yang, Jie Chen和 Yuexian Zou
# Github链接: https://github.com/zhang-can/CoLA

# 引入必要的库
import numpy as np
import os
from easydict import EasyDict as edict

# 定义配置文件
cfg = edict()

# GPU ID
cfg.GPU_ID = '0'

# 学习率
cfg.LR = '[0.0001]*6000'

# 迭代次数
cfg.NUM_ITERS = len(eval(cfg.LR))

# 类别数
cfg.NUM_CLASSES = 20

# 模态（数据类型）
cfg.MODAL = 'all'

# 特征维度
cfg.FEATS_DIM = 2048

# batch大小
cfg.BATCH_SIZE = 16

# 数据路径
cfg.DATA_PATH = './data/THUMOS14'

# 工作线程数
cfg.NUM_WORKERS = 8

# 正则项系数lambda
cfg.LAMBDA = 0.01

# 简单负样本数量
cfg.R_EASY = 5

# 困难负样本数量
cfg.R_HARD = 20

# 负样本数量比例因子m
cfg.m = 3

# 正样本数量比例因子M
cfg.M = 6

# 测试模型的频率
cfg.TEST_FREQ = 100

# 打印训练信息的频率
cfg.PRINT_FREQ = 20

# 类别阈值
cfg.CLASS_THRESH = 0.2

# NMS阈值
cfg.NMS_THRESH = 0.6

# CAS阈值（用于计算类别感知的时序动作定位性能）
cfg.CAS_THRESH = np.arange(0.0, 0.25, 0.025)

# ANESS阈值（用于计算类别感知的时序动作定位性能）
cfg.ANESS_THRESH = np.arange(0.1, 0.925, 0.025)

# TIOU阈值（用于计算时序动作定位性能）
cfg.TIOU_THRESH = np.linspace(0.1, 0.7, 7)

# 上采样因子
cfg.UP_SCALE = 24

# GT标注文件路径
cfg.GT_PATH = os.path.join(cfg.DATA_PATH, 'gt.json')

# 随机种子
cfg.SEED = 0

# 特征帧率
cfg.FEATS_FPS = 25

# 视频分段数目
cfg.NUM_SEGMENTS = 750

# 类别字典，用于将类别名称映射成数字
cfg.CLASS_DICT = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 
                  'CleanAndJerk': 3, 'CliffDiving': 4, 'CricketBowling': 5, 
                  'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 
                  'GolfSwing': 9, 'HammerThrow': 10, 'HighJump': 11, 
                  'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 
                  'Shotput': 15, 'SoccerPenalty': 16, 'TennisSwing': 17, 
                  'ThrowDiscus': 18, 'VolleyballSpiking': 19}

# 这段代码定义了一个配置文件`cfg`，包含了所有训练和测试过程中需要用到的参数设置。每个参数都有注释说明，方便用户根据自己的需求进行修改和调整。
