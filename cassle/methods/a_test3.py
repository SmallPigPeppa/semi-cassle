import torch
import numpy as np

def protoSave(self, model, loader, current_task):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, (indexs, images, target) in enumerate(loader):
            feature = model.feature(images.to(self.device))
            if feature.shape[0] == self.args.batch_size:
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
    labels_set = np.unique(labels)
    labels = np.array(labels)
    labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
    features = np.array(features)
    features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
    feature_dim = features.shape[1]

    prototype = []
    radius = []
    class_label = []
    for item in labels_set:
        index = np.where(item == labels)[0]
        class_label.append(item)
        feature_classwise = features[index]
        prototype.append(np.mean(feature_classwise, axis=0))
        if current_task == 0:
            cov = np.cov(feature_classwise.T)
            radius.append(np.trace(cov) / feature_dim)

    if current_task == 0:
        self.radius = np.sqrt(np.mean(radius))
        self.prototype = prototype
        self.class_label = class_label
        print(self.radius)
    else:
        self.prototype = np.concatenate((prototype, self.prototype), axis=0)
        self.class_label = np.concatenate((class_label, self.class_label), axis=0)

if __name__=='__main__':
    a={1:2,2:3}
    print(a[1])