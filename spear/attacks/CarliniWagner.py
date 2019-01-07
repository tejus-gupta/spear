import torch

class CarliniWagner():
    """
    The Carlini Wagner attack for generating adversarial examples as described in 'Towards Evaluating the
    Robustness of Neural Networks' by Carlini and Wagner. The attack produces closer adversarial examples
    but is slower than other attacks. 
    
    The default parameters are set to speedup the attack. You can find the parameters used in original paper
    here - https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py.
    
    Current implementation only supports targeted L2 attack. 
    """
    def __init__(self, binary_search_steps = 9, max_iters = 2000, confidence = 1, init_c = 0.01, lr = 0.01, abort_early = True, clip_min = 0, clip_max = 1, verbose = True):
        """
        :param binary_search_steps: number of binary search steps for finding best c
        :param max_iters: number of attack iterations
        :param confidence: encourages attack to find adversarial examples with high confidence of target class
        :param init_c: initial value of c
        :param lr: learning rate for gradient descent
        :param abort_early: If true, gradient descent is stopped early if no progress is made.
        :param clip_min: minimum value for clipping adversarial images
        :param clip_max: maximum value for clipping adversarial images
        :param verbose: If true, loss is printed at every max_iters/10 iterations.
        """
        self.binary_search_steps = binary_search_steps
        self.max_iters = max_iters
        self.confidence = confidence
        self.init_c = init_c
        self.lr = lr
        self.abort_early = abort_early
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.verbose = verbose
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def generate(self, net, images, labels = None):
        """
        Generates adversarial examples using Carlini and Wagner's targeted L2 attack.

        :param net: classification model
        :param images: clean images
        :param labels: target labels
        """
        def arctanh(x, eps=1e-6):
            x *= (1. - eps)
            return 0.5 * (torch.log((1 + x) / (1 - x)))
        
        with torch.no_grad():
            labels = labels.long()
            clean_images = images.clone()
            
            batch_size = images.shape[0]
            batch_idxs = torch.tensor(range(batch_size), device = images.device).long()

            lower_bound = torch.zeros((batch_size), device = images.device)
            upper_bound = 1e10 * torch.ones((batch_size), device = images.device)
            c = self.init_c * torch.ones((batch_size), device = images.device)

            min_l2 = 1e10 * torch.ones((batch_size), device = images.device)
            min_l2_images = torch.zeros(images.shape, device = images.device)
        
        for _ in range(self.binary_search_steps):
            with torch.no_grad():
                w = torch.nn.Parameter(arctanh(2 * images - 1))
            
            w.requires_grad = True
            optimizer = torch.optim.Adam([w], lr=self.lr)
            prev_loss = torch.tensor([float("inf")], device = images.device)

            success = torch.zeros((batch_size), device = images.device).byte()

            for iter_idx in range(self.max_iters):
                adv_images = 0.5 * (torch.tanh(w) + 1)
                outputs = net(adv_images)
                
                # f = sum_over_batches(outputs[highest_other_than_label] - outputs[label] + self.confidence)
                outputs2 = outputs.clone().detach()
                outputs2[batch_idxs, labels] = -float("inf")
                max_score_labels = torch.argmax(outputs2, 1)
                f = outputs.gather(1, max_score_labels.long().view(-1,1)) - outputs.gather(1, labels.long().view(-1,1)) + self.confidence
                f = f.squeeze(1)
                
                l2 = torch.pow(adv_images - clean_images, 2).sum(3).sum(2).sum(1)
                loss = l2.sum() + torch.sum(c * f)

                loss.backward()
                
                if self.verbose and iter_idx % (self.max_iters//10) == 0:
                    print(iter_idx, loss.item())
                
                if self.abort_early and iter_idx % (self.max_iters//10) == 0:
                    if loss > prev_loss * 0.9999:
                       break
                    prev_loss = loss.detach()
                
                success = torch.max(success, f<=0)
                min_l2_images[torch.min(l2<min_l2, f<=0),:,:] = adv_images[torch.min(l2<min_l2, f<=0),:,:]
                min_l2[f<=0] = torch.min(min_l2[f<=0], l2[f<=0])

                optimizer.step()
                optimizer.zero_grad()
            
            with torch.no_grad():
                upper_bound[success] = torch.min(upper_bound[success], c[success])
                lower_bound[1-success] = torch.max(lower_bound[1-success], c[1-success])

                c[upper_bound < 1e9] = (lower_bound[upper_bound < 1e9] + upper_bound[upper_bound < 1e9])/2
                c[upper_bound > 1e9] = 10 * c[upper_bound > 1e9]
        
        return min_l2_images



