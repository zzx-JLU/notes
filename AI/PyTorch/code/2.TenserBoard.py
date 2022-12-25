from torch.utils.tensorboard import SummaryWriter

# SummaryWriter 类用于向指定目录下的事件文件写入数据，供 TenserBoard 使用
# 创建 SummaryWriter 对象时，需要指定文件夹的路径。如果不指定，默认为 runs/**CURRENT_DATETIME_HOSTNAME**
writer = SummaryWriter("logs")

# add_scalar() 方法用于添加标量数据
#   - tag: 标题
#   - scalar_value: 要保存的标量值，作为 y 轴
#   - global_step: 步骤数，作为 x 轴
for i in range(100):
    writer.add_scalar('y=2x', 2 * i, i)

# 使用完毕后需要关闭
writer.close()
