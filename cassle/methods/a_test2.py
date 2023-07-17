import torch
import random
import torch.nn as nn

if __name__ == '__main__':
    num_class = 100
    num_tasks = 5
    old_class = []
    tasks = torch.randperm(num_class).chunk(num_tasks)
    current_task = 0
    for task in tasks[:current_task]:
        old_class.extend(task.tolist())
    radius = 0.2
    features_dim = 2
    prototype = nn.ParameterList(
        [nn.Parameter(torch.randn(1, features_dim)) for i in range(num_class)])
    batch_size = 16
    device = 'cpu'
    # index 应该从old class中采样batchsize个
    old_y = torch.tensor(random.choices(old_class, k=batch_size)).to(device)

    # 将old_y转化为Python的整数列表
    old_y_list = old_y.tolist()

    # 使用old_y_list去索引prototype
    old_x = torch.cat([prototype[i] for i in old_y_list])

    old_x = torch.cat([prototype[i] for i in old_y_list]) + torch.randn(batch_size, 2).to(device) * radius
    print(old_x, old_y)
