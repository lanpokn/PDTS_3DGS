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
- **RiskLearner**: Neural Process架构 (x_dim=10, y_dim=1, r_dim=10, z_dim=10, h_dim=10)

#### **2. Training Logic** (复制自PDTS trainer):
**来源**: `PDTS/sinusoid/Model/trainer_risklearner.py`
- **Loss函数**: MSE reconstruction loss + KL divergence regularization  
- **Prior更新**: 用变分后验更新先验分布 (Line 164-166)
- **训练频率**: 每num_selected_views轮训练一次 (模仿PDTS的每epoch训练)

#### **3. View Selection Strategy** (改编自PDTS采样):
**来源**: `PDTS/sinusoid/trainer_maml.py` Line 178-199 ("diverse"分支)
- **Posterior Sampling**: 用预测均值μ作为acquisition score (Line 94)
- **Diversity Regularization**: Maximum Sum of Distances (MSD) 算法
- **Bootstrap Phase**: 前200轮随机采样收集初始数据

#### **4. Camera Feature Extraction** (新设计):
**设计原理**: 将3DGS相机参数转换为网络输入特征
```python
# 10维特征向量:
- camera_center (3D): 世界坐标中的相机位置
- euler_angles (3D): 旋转矩阵R转换为欧拉角表示朝向  
- fov (2D): FoVx, FoVy视场角
- resolution (2D): 图像宽高
```

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
**Current status**: ✅ PDTS integration完成，支持bootstrap → network prediction transition
**Key achievement**: 用Neural Process预测view difficulty，避免昂贵的3DGS forward pass
**Next**: 测试性能对比，验证PDTS是否真正加速了training