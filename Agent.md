# PLF - 蒙西节点电价预测与储能策略优化

基于深度学习与机器学习的电价时序预测项目，并集成储能充放电策略优化。

---

## Goal

构建蒙西地区电力现货市场节点实时电价预测系统（15min 间隔，预测 D+1 日 96 个点），并据此制定储能充放电策略以最大化收益。

---

## Constraints & Preferences

### 运行环境

- **conda env**: `TSF`（PyTorch 2.11.0+cu128）
- **工作目录**: `D:\YZH\Item\PowerLoadForcast`

### 数据规则

- 滚动统计必须后向窗口，避免数据泄露
- 数据划分按1-10月训练，11月验证，12月测试
- 归一化仅用 train 集统计量(min-max)
- 测试集不提供边界条件中的实际值，仅使用预测值列

### 评测标准

- **储能容量**: 8000 kWh，充放电功率 ±1000 kW
- **操作约束**: 每日最多一次完整充放电，连续 8 个时间点（2h），且操作必须在当天完成
- **收益公式**: Profit = Σ(P_t × E_t)，最大化测试集日均收益

---

## 项目架构

```
PowerLoadForcast/
├── docs/
│   └── 赛题.md               # 赛题说明
├── config/                   # 模型配置文件
│   └── {model}.yml
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据
│   │   ├── train/            # 训练集（边界条件 + 电价）
│   │   ├── test/             # 测试集（边界条件）
│   │   └── all_nc/           # NWP 气象预报（nc 格式）
│   └── proc/                 # 预处理后数据
├── dataset/                  # 数据集构建
│   └── mengxi.py
├── model/                    # 完整模型定义
│   └── {model}.py
├── module/                   # 模型子模块
│   └── {model}.py
├── trainer/                  # 训练器
│   ├── trainer_base.py
│   └── trainer_{model}.py
├── tool/                     # 工具集
│   ├── early_stopping.py
│   ├── logger.py
│   ├── metrics.py
│   ├── strategy.py           # 储能策略优化
│   └── utils.py
├── output/                   # 输出目录
│   ├── checkpoint/           # 模型权重
│   ├── image/                # 可视化图表
│   ├── log/                  # 训练日志
│   ├── output_demo.csv       # 提交格式示例
│   └── result/               # 预测结果
├── Agent.md                  # 项目架构与职责定义
├── data_engineer.py          # 数据工程（清洗、对齐、nc解析）
├── feature_engineer.py       # 特征工程（衍生特征、时间特征）
├── lgb_baseline.py           # LightGBM 基线（端到端脚本）
├── train.py                  # 训练入口
├── test.py                   # 测试入口（推理 + 策略生成）
```

---

## 目录职责

### `data/` — 数据存储

存放原始数据与处理后数据，按处理阶段划分子目录。

- `raw/train/`: 训练集原始数据
  - `mengxi_boundary_anon_filtered.csv`: 蒙西边界条件（系统负荷、
    风光总加、联络线、风电、光伏、水电、非市场化机组的实际值/预测值）
  - `mengxi_node_price_selected.csv`: 节点实时电价（A 列）
- `raw/test/`: 测试集原始数据
  - `test_in_feature_ori.csv`: 测试集边界条件（仅含预测值）
- `raw/all_nc/`: NWP 气象预报数据（NetCDF 格式，7 个变量 × 逐小时）
- `proc/`: 经 `data_engineer.py` 处理后的中间数据
- 仅负责存储，不包含代码逻辑

### `data_engineer.py` — 数据工程

负责原始数据的加载、清洗与对齐。

- 加载 CSV 边界条件数据与节点电价数据
- 解析 NetCDF 气象数据（u100/v100/t2m/tp/tcc/sp/ghi）
- NetCDF的世界时转化为北京时
- 每个nc文件为发布日对应的未来1天逐小时预报。例如 `20240101.nc` 表示 2024年 1月 1日发布、对 1月 2日的逐小时气象预测。
- 处理缺失值、异常值
- nc文件时间分辨率为1h，csv文件时间分辨率为15min，nc文件需要插值为15分钟进行时间对齐
- 将清洗后的数据存入 `data/proc/`

**输入**: `data/raw/` 中的原始文件
**输出**: `data/proc/` 中的清洗数据，供 `dataset/` 使用

### `feature_engineer.py` — 特征工程(可选, 考虑深度学习模型是否需要构建额外特征)

负责在清洗数据基础上构建衍生特征，是数据管线的第二阶段(可选)。

- 时间特征（hour、minute、dayofweek、month等）
- 统计特征（滚动均值、极差、标准差等，仅后向窗口）
- ...etc

**输入**: `data/proc/` 中的清洗数据
**输出**: 带完整特征的 csv文件，供 `dataset/` 使用

### `dataset/` — 数据集构建

负责滑动窗口构建与 DataLoader 生成，将 `data/proc/` 中的 csv文件 转为模型可用输入。

- `mengxi.py`:
  - 加载"数据工程后"的 csv文件 or "特征工程后"的 csv文件(如果有额外特征)
  - 滑动窗口切分（`seq_len` → `pred_len`）
  - `Dataset` 类实现
  - `get_dataloader()` 工厂函数，返回 train/val/test 的 DataLoader

**输入**: `feature_engineer.py` or `data_engineer.py` 输出的 csv文件
**输出**: 可直接送入模型的 DataLoader

### `model/` — 完整模型

每个文件定义一个完整的可训练模型，负责将子模块组装成端到端的前向传播流程。

- 每个模型类应实现 `__init__()` 和 `forward()` 方法
- `forward()` 接收原始输入（如 `x_enc`），返回预测结果
- 可直接实例化为 `nn.Module` 送入训练器

**依赖**: `module/` 中的子模块

### `module/` — 模型子模块

存放各模型的核心组件和基础构建块，如注意力机制、嵌入层、MLP 块等。

- 与 `model/` 一一对应，每个文件为同名模型提供子模块
- 每个子模块应职责单一、可独立测试
- 避免跨模型的耦合依赖

### `trainer/` — 训练器

训练逻辑的核心，采用基类 + 派生类的设计模式。

- `trainer_base.py`:
  - 训练基类，定义标准训练流程（epoch 循环、前向传播、损失计算、
    反向传播、梯度裁剪等）
  - 提供 `train_batch()`、`train_epoch()`、`evaluate_batch(mode="valid"/"test"，验证只看loss，test需要指标)`、
    `train()`、`valid(调用evaluate_batch(mode=valid))`、`test(调用evaluate_batch(mode=test))`、
    `get_optimizer()`、`get_scheduler()` 等通用接口
  - 预定义多种优化器（Adam、AdamW、SGD 等）和学习率调度器
    （CosineAnnealingLR、StepLR、ReduceLROnPlateau 等），通过配置选择
- `trainer_{model}.py`:
  - 继承基类，实现特定模型的训练细节
  - 可覆写损失函数、前向逻辑、特殊调度策略等

**依赖**: `model/`、`tool/`（logger、early_stopping、metrics）

### `tool/` — 工具集

提供训练和评估过程中所需的通用工具。

| 文件 | 职责 |
|------|------|
| `logger.py` | 日志记录（训练过程、配置信息、系统输出） |
| `metrics.py` | 评估指标计算（RMSE、MAE、R²） |
| `early_stopping.py` | 早停机制（监控验证集指标，支持 min/max 模式） |
| `strategy.py` | 储能充放电策略优化（基于预测电价搜索最优充放电时段） |
| `utils.py` | 辅助函数（随机种子设置、配置加载等） |

### `config/` — 配置文件

每个模型对应一个 YAML 配置文件，包含：

- 模型超参数（`seq_len`、`pred_len`、`d_model`、`n_heads` 等）
- 训练超参数（`lr`、`batch_size`、`epochs` 等）
- 数据路径与预处理参数
- 每个超参数带有注释说明便于调参理解

### `output/` — 输出目录

按类型组织所有运行产物。

- `checkpoint/`: 训练保存的最佳模型权重
- `image/`: 可视化图表（预测曲线、loss 曲线等）
- `log/`: 训练日志文件
- `result/`: 预测结果 CSV（含电价预测与充放电策略）

### `lgb_baseline.py` — LightGBM 基线

独立端到端脚本，提供基于 LightGBM 的电价预测与策略生成基线。

- 使用边界条件预测值列 + 时间特征作为输入
- 按时间顺序划分 1-10月数据为训练集，11月数据为验证集，12月数据为测试集
- 包含充放电策略生成框架（`generate_strategy()`）
- 作为深度学习方案的对照基准

### `train.py` — 训练入口

训练脚本的唯一入口，负责：

1. 解析命令行参数或加载配置文件
2. 构建数据集和 DataLoader
3. 实例化模型和训练器
4. 执行训练流程
5. 保存最佳模型权重至 `output/checkpoint/`
6. 保存训练日志至 `output/log/`

### `test.py` — 测试入口

测试脚本的唯一入口，负责：

1. 加载训练好的最佳模型权重
2. 在测试集上执行推理，生成 D+1 日 96 点电价预测
3. 调用 `tool/strategy.py` 生成最优充放电策略
4. 输出符合赛题格式的 CSV 至 `output/result/`
5. 生成可视化结果至 `output/image/`

---

## 数据流

```
data/raw/ → data_engineer.py → data/proc/
                                   ↓
                     feature_engineer.py（特征构建 可选）
                                   ↓
                     dataset/（滑动窗口 + DataLoader）
                                   ↓
                     model/ → trainer/ → train.py → output/checkpoint/
                                ↓                      ↓
                     tool/metrics              test.py → output/result/
                     tool/strategy                      → output/image/
```

---

## 编码规范

- **设计**: 模块化，单一职责，简洁可扩展
- **命名**: 遵循 PEP 8（类 PascalCase、方法/变量 snake_case）
- **注释**: 使用中文注释，docstring 控制在三行以内
- **缩进**: 4 空格，行宽不超过 80 字符
- **导入顺序**: 标准库 → 第三方库 → 本地模块
- **模型注册**: 新增模型时需同步创建 `config/{name}.yml`、
  `module/{name}.py`、`model/{name}.py`、`trainer/trainer_{name}.py`
