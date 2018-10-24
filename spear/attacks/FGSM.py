import torch

class FGSM():
    """
    The Fast Gradient Sign Method of generating adversarial examples as described in 'Explaining and harnessing
    adversarial examples' by Goodfellow et al.
    """
    def __init__(self, eps = 0.3):
        """
        :param eps: attack step size
        """
        self.eps = eps
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def generate(self, net, images, labels = None):
        """
        Generates adversarial examples using the Fast Gradient Sign Method.

        :param net: classification model
        :param images: clean images
        :param labels: (optional) labels of clean images. If the labels of clean images aren't given,
        the network's predictions are used as labels. This is useful for avoiding the label leaking effect.
        See 'Adversarial machine learning at scale' by Kurakin et al. for details about the label
        leaking effect.
        """
        images.requires_grad = True

        if images.grad is not None:
            images.grad.data.zero_()

        net.zero_grad()
        
        outputs = net(images)
        if labels is None:
            loss = self.criterion(outputs, torch.max(outputs.data, 1)[1])
        else:
            loss = self.criterion(outputs, labels)
        loss.backward()
        
        with torch.no_grad():
            return torch.clamp(images + self.eps * torch.sign(images.grad), 0, 1)