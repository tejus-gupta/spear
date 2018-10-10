import torch

class FGSM():
    def __init__(self, eps = 0.3):
        self.eps = eps
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def attack(self, net, images, labels = None):
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