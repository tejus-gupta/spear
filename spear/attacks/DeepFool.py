import torch

class DeepFool():
    """
    The DeepFool algorithm for generating adversarial examples as described in 'DeepFool: a simple and
     accurate method to fool deep neural networks' by Moosavi-Dezfooli et al.
    """
    def __init__(self, norm = 2, max_iter = 15, step_size = 1.02):
        """
        :param norm: order of norm used for measuring perturbation (default is 2)
        :param max_iter: maximum number of iterations
        :param step_size: In order to cross the classification boundary, the final perturbation is multiplied
        by step_size. (default is 1.02)
        """
        self.norm = norm
        self.max_iter = max_iter
        self.step_size = step_size
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def generate(self, net, images, labels):
        """
        Generates adversarial examples using the DeepFool algorithm.

        :param net: classification model
        :param images: clean images. batch size must be one.
        """

        assert images.shape[0] == 1, "Current implementation of DeepFool only supports batch size of one!"

        label = labels.item()

        if self.norm != float("inf"):
            q = self.norm / (self.norm-1.0)

        with torch.no_grad():
            clean_images = images.clone()
        
        dist = 0

        for iter_idx in range(self.max_iter):
            
            images.requires_grad = True
            outputs = net(images)

            if torch.max(outputs, 1)[1] != label:
                return images
            
            min_dist = float("inf")
            
            for k in range(outputs.shape[1]):
                if k == label:
                    continue

                if images.grad is not None:
                    images.grad.data.zero_()
                
                (outputs[0][k] - outputs[0][label]).backward(retain_graph = True)
                w_k = images.grad
                f_k = outputs[0][k] - outputs[0][label]

                if self.norm == float("inf"):
                    if abs(f_k)/torch.norm(w_k, 1) < min_dist:
                        min_dist = abs(f_k)/torch.norm(w_k, 1)
                        w_l = w_k
                        f_l = f_k
                else:
                    if abs(f_k)/torch.norm(w_k, q) < min_dist:
                        min_dist = abs(f_k)/torch.norm(w_k, q)
                        w_l = w_k
                        f_l = f_k

            with torch.no_grad():
                if self.norm == float("inf"):
                    dist += min_dist
                    images = torch.clamp(images + min_dist * torch.sign(w_l), 0, 1)
                else:
                    images = torch.clamp(images + abs(f_l) / torch.pow(torch.norm(w_l, q), q) * torch.pow(torch.abs(w_l), q-1) * torch.sign(w_l), 0, 1)
        
        with torch.no_grad():
            return clean_images + self.step_size * (images - clean_images)