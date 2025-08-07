# Claude Memory for 3DGS + Active Sampling Project

## 🎯 ULTIMATE PROJECT GOAL
**Replace Direct Loss Calculation with Neural Network Prediction**
- **Problem**: Computing loss for all views via 3DGS forward pass is too time-consuming
- **Solution**: Train a neural network that takes views as input and predicts loss as output
- **Method**: Adapt PDTS's risk learner approach to 3DGS view selection
- **Architecture**: Network(view) → predicted_loss, then use PDTS sampling strategies

## Project Overview
This is a dual project combining:
1. **3DGS (3D Gaussian Splatting)** - Famous real-time radiance field rendering implementation  
2. **PDTS (Posterior and Diversity Synergized Task Sampling)** - Novel robust active task sampling research project
3. **Integration Goal**: Use PDTS's risk prediction network to accelerate 3DGS view selection

## 3DGS Main Project - MODIFIED WITH LOSS-BASED VIEW SELECTION
- **Core Implementation**: Standard 3D Gaussian Splatting for real-time rendering
- **Key Files**: train.py (MODIFIED), render.py, scene/, utils/, gaussian_renderer/
- **Status**: Enhanced with intelligent view selection algorithm

### CURRENT IMPLEMENTATION: Direct Loss-Based View Selection (FIXED)
**Implementation Details:**
- **Flag**: `--loss_judge` enables the current algorithm
- **Parameter**: `--num_selected_views` (default: 16) - both number of views to select AND training interval
- **Algorithm**: Every N iterations, randomly select 32 candidates, evaluate their loss, select top N with highest loss
- **Current Usage**: `--num_selected_views 4` (32选4策略)
- **Command**: `python train.py -s <dataset> -m <output> --iterations 2000 --loss_judge --num_selected_views 4`
- **Key Fix**: Changed from "evaluate ALL views" to "evaluate 32 random candidates" to maintain randomness
- **Limitation**: Still requires 3DGS forward pass for candidates → SLOW but more reasonable

### ✅ IMPLEMENTED: PDTS Neural Network Integration 
**Code Location**: `pdts_integration.py` + modifications in `train.py`

#### **1. Core Neural Network Components** (直接复制自PDTS源码):
**来源**: `PDTS/sinusoid/Model/risklearner.py` + `PDTS/sinusoid/Model/backbone_risklearner.py`
- **Encoder**: Maps (camera_features, loss) → representation r_i
- **MuSigmaEncoder**: Aggregates representations → μ,σ for latent variable z  
- **Decoder**: Maps (camera_features, z) → predicted loss
- **RiskLearner**: Neural Process架构 (x_dim=13, y_dim=1, r_dim=10, z_dim=10, h_dim=10) - 已更新维度

#### **2. Training Logic** (复制自PDTS trainer):
**来源**: `PDTS/sinusoid/Model/trainer_risklearner.py`
- **Loss函数**: MSE reconstruction loss + KL divergence regularization  
- **Prior更新**: 用变分后验更新先验分布 (Line 164-166)
- **训练频率**: 每num_selected_views轮训练一次 (模仿PDTS的每epoch训练)

#### **3. View Selection Strategy** (IMPROVED 已修复):
**来源**: `PDTS/sinusoid/trainer_maml.py` Line 178-199 ("diverse"分支)
- **Posterior Sampling**: 真正的随机采样 (返回单个stochastic prediction而非均值)
- **Diversity Regularization**: 修复的MSD算法，正确的归一化和权重平衡
- **Bootstrap Phase**: 前200轮随机采样收集初始数据

**Critical Fixes Applied:**
- ✅ **Fixed Posterior Sampling**: `predict_for_sampling`现在返回单个随机样本而非确定性均值
- ✅ **Fixed Diversity Scoring**: 实现Min-Max归一化，确保acquisition_score和diversity_score在相同尺度[0,1]
- ✅ **Added Trade-off Parameter**: `lambda_diversity`参数实现真正的平衡: `(1-λ)*acquisition + λ*diversity`

#### **4. Camera Feature Extraction** (IMPROVED 已修复):
**设计原理**: 将3DGS相机参数转换为网络输入特征
```python
# 13维特征向量 (已改进):
- camera_center (3D): 世界坐标中的相机位置
- rotation_6d (6D): 6D旋转表示 (替代欧拉角，避免万向锁)
- fov (2D): FoVx, FoVy视场角
- resolution (2D): 图像宽高
```
**Key Fixes:**
- ✅ 替换欧拉角为6D旋转表示，避免万向锁和不连续性
- ✅ 添加tensor类型检查，支持numpy array输入
- ✅ 更新x_dim=13以匹配新特征维度

#### **5. Integration with 3DGS Training Loop**:
**设计原理**: 最小化对原train.py的修改
- **数据收集**: 每轮训练后添加(camera_features, actual_loss)到训练数据
- **网络更新**: 每selection_interval轮训练RiskLearner 
- **UI集成**: PDTS网络loss显示在进度条"PDTS Loss"字段

### Key Components to Borrow from PDTS:
- **RiskLearner Architecture** (Neural Process-based)
- **Posterior Sampling** mechanism  
- **Diversity Regularization** (MSD algorithm)
- **Training Loop** for risk predictor updates

## PDTS Project (Located in PDTS/ folder)
- **Purpose**: Robust active task sampling for adaptive decision-making
- **Core Algorithm**: Combines posterior sampling + diversity regularization
- **Domains**: 
  - sinusoid/ - Regression experiments with MAML
  - RL/DR/ - Reinforcement learning with domain randomization

### Key PDTS Components
1. **RiskLearner**: Neural Process-based risk predictor (uncertainty-aware)
2. **Sampling Strategies**: ERM, DRM, MPTS, PDTS
3. **Diversity Regularization**: Maximum Sum of Distances (MSD)
4. **Acquisition Function**: γ_μ × μ + γ_σ × σ (exploration vs exploitation)

### PDTS Algorithm Flow
1. Generate candidate tasks
2. Fast difficulty evaluation via risk predictor
3. Select diverse + challenging task subset
4. Train policy on selected tasks
5. Update risk predictor with true performance

## 🔄 INTEGRATION STRATEGY

### Technical Adaptation Challenges:
1. **Input Representation**: 
   - PDTS: Task parameters (amplitude, phase for sinusoid; env params for RL)
   - 3DGS: View parameters (camera pose, intrinsics, etc.)
   - **Solution**: Extract meaningful view features as network input

2. **Output Scaling**:
   - PDTS: Task difficulty scores
   - 3DGS: Rendering loss values
   - **Solution**: Normalize/calibrate loss predictions

3. **Training Data**:
   - PDTS: Meta-learning across different tasks
   - 3DGS: Single scene, multiple views
   - **Solution**: Adapt training loop for single-scene view prediction

### Implementation Plan:
1. **Phase 1**: Create `pdts_integration.py` with basic risk learner
2. **Phase 2**: Extract view features from camera parameters
3. **Phase 3**: Train predictor using current direct loss data
4. **Phase 4**: Replace direct calculation with prediction in `train.py`
5. **Phase 5**: Add diversity regularization and posterior sampling

### Architecture Design:
```
View Features → Risk Learner Network → Predicted Loss → PDTS Sampling → Selected Views
     ↑                                        ↓
Camera Params                           Ground Truth Loss
(pose, intrinsics)                    (for network training)
```

## Testing and Datasets
- **datasets/** folder contains test data (tandt_db with drjohnson, playroom, train, truck scenes)
- **test_loss_judge.py** - automated testing script comparing baseline vs loss-based selection
- **Test Results**: Stored in test_results/ directory with JSON format

## Important Notes
- PDF paper couldn't be read, but Chinese summary provided good understanding
- Code analysis shows high-quality modular implementation
- PDTS implements multiple baseline methods for comparison
- **CURRENT STATUS**: 32选4 direct loss calculation working, ready for PDTS integration
- **NEXT PRIORITY**: Implement `pdts_integration.py` with bootstrap strategy
- **DESIGN PRINCIPLE**: Keep train.py minimal, put PDTS integration logic in separate file
- **Bootstrap Strategy**: Start with random sampling to collect initial training data, then switch to network prediction
- **Code Strategy**: Copy/reuse PDTS source code for complex mathematical components
- User runs tests on Windows, code modifications done in WSL2

### 📋 PDTS Code Mapping (Implementation完成后的对照表):

#### **PDTS源码 → 3DGS集成代码映射**:
- `PDTS/sinusoid/Model/backbone_risklearner.py` → `pdts_integration.py` (Encoder, MuSigmaEncoder, Decoder)
- `PDTS/sinusoid/Model/risklearner.py` → `pdts_integration.py` (RiskLearner class)
- `PDTS/sinusoid/Model/trainer_risklearner.py` → `pdts_integration.py` (RiskLearnerTrainer class)
- `PDTS/sinusoid/trainer_maml.py` Line 178-199 → `pdts_integration.py` (msd_diversified_score)
- `PDTS/sinusoid/trainer_maml.py` Line 173,199 → `train.py` Line 240 (training frequency logic)

#### **运行命令**:
```bash
# PDTS模式
python train.py -s ./datasets/tandt_db/tandt/truck -m ./output/pdts_test --iterations 1000 --pdts --num_selected_views 4

# 对比baseline  
python train.py -s ./datasets/tandt_db/tandt/truck -m ./output/baseline_test --iterations 1000
```

## 📋 PROJECT SUMMARY  
**What we had**: 3DGS with 32选4 direct loss calculation (slow but working)
**What we implemented**: PDTS neural network integration for fast view difficulty prediction
**Current status**: ✅ PDTS integration完成并修复关键bug，支持完全可配置的参数
**Key achievement**: 用Neural Process预测view difficulty，避免昂贵的3DGS forward pass

### 🔧 **CRITICAL FIXES COMPLETED**:
1. **Diversity Scoring Algorithm**: 修复diversity和acquisition score的尺度不匹配问题
   - 实现Min-Max归一化确保两个score在[0,1]范围内
   - 添加`lambda_diversity`平衡参数实现proper trade-off
   
2. **Posterior Sampling Mechanism**: 修复随机性缺失问题  
   - `predict_for_sampling`现在返回真正的stochastic sample
   - 符合PDTS paper Eq.12的Thompson Sampling理论要求
   
3. **Camera Feature Representation**: 提升特征稳定性
   - 替换欧拉角为6D旋转表示避免万向锁
   - x_dim更新为13维，添加robust tensor处理
   
4. **Code Robustness**: 处理数据类型兼容性
   - 支持numpy array和tensor混合输入
   - 防止`AttributeError`运行时错误

### 🛠️ **LATEST UPDATES** (2025-08-07):
5. **Full Parameter Configurability**: 所有关键参数现在都支持命令行配置
   - ✅ `--num_selected_views` (默认4): 最终选择的视角数量
   - ✅ `--num_candidate_views` (默认32): 候选视角池大小
   - ✅ `--lambda_diversity` (默认0.5): exploitation vs exploration权衡参数
   
6. **Improved Default Values**: 更新为更实用的默认配置
   - 从16选4改为32选4策略 (更合理的候选池)
   - λ=0.5实现balanced exploration-exploitation
   
7. **Parameter Validation**: 确认热身阶段功能正常
   - ✅ Bootstrap期间PDTS正常收集训练数据但不参与view selection
   - ✅ 两种模式(loss_judge + PDTS)都正确选择高loss视角
   - ✅ λ=0.0时退化为pure exploitation模式 (符合预期)

8. **🚨 CRITICAL PROBLEM IDENTIFIED & FIXED**: PDTS模型在10000轮附近崩溃
   - **根本原因**: 3DGS每3000轮进行opacity reset，导致scene突变，PDTS预测模型过拟合历史数据
   - **解决方案**: Periodic Recovery Sampling Strategy
     - 每次opacity reset后的500轮自动切换回随机采样
     - 数据收集继续进行，帮助模型适应新的loss分布
     - 500轮后恢复网络预测模式
   - **实现细节**: 
     - `_is_in_recovery_period()`: 检测是否在恢复期
     - 自动适配`opt.opacity_reset_interval` (默认3000)
     - Recovery period可配置 (默认500轮)

**Command Examples**:
```bash
# 默认PDTS (32选4, λ=0.5)  
python train.py -s dataset -m output --pdts

# 自定义参数
python train.py -s dataset -m output --pdts --num_selected_views 8 --num_candidate_views 64 --lambda_diversity 0.3

# Pure exploitation (λ=0.0)
python train.py -s dataset -m output --pdts --lambda_diversity 0.0
```

**Timeline of PDTS Behavior**:
```
轮次:    0-2000      2001-3000    3001-3500    3501-6000    6001-6500    6501-9000    9001-9500    ...
模式:    bootstrap   network      recovery     network      recovery     network      recovery     ...
原因:    初始训练    正常预测     opacity      正常预测     opacity      正常预测     opacity      ...
                               reset                    reset                    reset
```

**Status**: ✅ Overfitting问题已解决，ready for robustness testing