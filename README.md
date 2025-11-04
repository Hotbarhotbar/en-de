# 基于 Pytorch 从零实现 Transformer (Mid-Term Assignment)

本项目根据 [大模型] 的要求，从零开始实现了一个完整的 Transformer 模型，并在 IWSLT 2017 (en-de) 数据集上进行了训练和消融实验。

## 1. 目录结构

"""
/
├── configs/
│   └── improved_25plus.yaml    # 最终版超参数配置
├── data/                       # 存放 en-de 数据 [cite: 22]
├── results/
│   └── ablation_results.csv    # 最终消融实验结果表格 [cite: 15]
│   └── run_baseline/
│       └── 20251102_225435/
│           ├── train.log
│           ├── train_log.csv
│           ├── train_valid_loss.png # 训练曲线图 [cite: 15]
│           └── best_model.pt      # 最佳模型
│ 
├── run_ablation_improved.sh  # 
├── src/
│   ├── layers.py               # MHA, FFN, PositionalEncoding 等
│   ├── model_improved.py       # Transformer 完整模型  
│   ├── train_improved.py       # 训练脚本
│   ├── evaluate_improved.py    # 评估脚本 (BLEU)
│   ├── dataset.py              # 数据加载
│   └── ...
├── requirements.txt            # 依赖
└── README.md                   # 本文件

"""
数据压缩包在data/raw下 解压到raw下 运行prepare_data.py即可获得处理好的数据
在目录结构文件下直接运行run.sh可以获得baseline 直接运行run_ablation_improved.sh 可以获得baseline和消融实验的结果
