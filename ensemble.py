import numpy as np

# 加载模态数据
modality1 = np.load(r'work_dir/uav/csub/bone_test/epoch1_test_score.npy') 
modality2 = np.load(r'work_dir/uav/csub/joint/epoch1_test_score.npy')  
# 如果有更多模态，继续加载
# 设定权重
weights = [0.5, 0.5]  # 模态1权重0.5，模态2权重0.5
# 计算加权平均
fused_output = weights[0] * modality1 + weights[1] * modality2
# 保存结果
np.save('pred.npy', fused_output)
print("融合完成，置信度文件已保存为 'pred.npy'")
