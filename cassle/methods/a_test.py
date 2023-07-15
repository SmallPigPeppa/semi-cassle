import torch
import random
if __name__ == '__main__':
    old_class = [0, 1, 2, 3]
    radius = 0.2
    prototype = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    batch_size = 16
    device = 'cpu'
    # index 应该从old class中采样batchsize个
    old_y = torch.tensor(random.choices(old_class, k=batch_size)).to(device)
    old_x = prototype[old_y] + torch.randn(batch_size, 2).to(device) * radius
    print(old_x,old_y)


