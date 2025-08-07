  在训练过程中的指定迭代次数，代码会：
  1. 使用scene.getTestCameras()获取测试视角
  2. 对每个测试视角进行渲染
  3. 计算PSNR和L1 loss指标
  4. 在tensorboard中记录测试结果

  结论: 您的train数据集使用LLFF Hold策略，每8张图片选1张作为测试视角，总共约38个测试视角用于验证训练效果。

> 请你简单回复一下，测试集的图片会出现在训练集中吗？

● 不会。

  根据代码逻辑：
  - 训练集：[c for c in cam_infos if not c.is_test] (非测试视角)
  - 测试集：[c for c in cam_infos if c.is_test] (测试视角)

  测试集和训练集是完全分离的，测试集的图片不会出现在训练过程中。