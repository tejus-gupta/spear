import torch

class CarliniWagner():
    """
    The Carlini Wagner attack for generating adversarial examples as described in 'Towards Evaluating the
    Robustness of Neural Networks' by Carlini and Wagner. This method only supports targeted attack.
    """
    def __init__(self, eps = 0.3, step_size = 0.05, iter = 10, targeted = False, clip_min = 0, clip_max = 1):
        """
        :param eps: attack step size
        :param step_size: step size for each iteration of attack
        :param iter: number of attack iterations
        :param targeted: whether to use targeted attack
        :param clip_min: minimum value for clipping adversarial images
        :param clip_max: maximum value for clipping adversarial images
        """
        self.eps = eps
        self.step_size = step_size
        self.iter = iter
        self.targeted = targeted
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def generate(self, net, images, labels = None):
        """
        Generates adversarial examples using the Fast Gradient Sign Method.

        :param net: classification model
        :param images: clean images
        :param labels: (optional) For untargeted attack, labels of clean images can be provided. If the labels
        of clean images aren't given, the network's predictions are used as labels. This is useful for avoiding
        the label leaking effect. See 'Adversarial machine learning at scale' by Kurakin et al. for details
        about the label leaking effect. For targeted attack, target labels can be provided. If target labels aren't
        given, least likely classes are used as target.
        """
        with torch.no_grad():
            clean_images = images.clone()
        
        for iter_idx in range(self.iter):
            images.requires_grad = True

            if images.grad is not None:
                images.grad.data.zero_()
            
            outputs = net(images)

            if self.targeted:
                if labels is None:
                    loss = -self.criterion(outputs, torch.min(outputs.data, 1)[1])
                else:
                    loss = -self.criterion(outputs, labels)
            else:
                if labels is None:
                    loss = self.criterion(outputs, torch.max(outputs.data, 1)[1])
                else:
                    loss = self.criterion(outputs, labels)

            loss.backward()
            
            with torch.no_grad():
                images = torch.clamp(images + self.eps * torch.sign(images.grad), self.clip_min, self.clip_max)
                images = torch.max(images, clean_images - self.eps)
                images = torch.min(images, clean_images + self.eps)
        
        return images