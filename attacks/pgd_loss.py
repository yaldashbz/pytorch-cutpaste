import torch


class LossPGDAttack:
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
    
    def __call__(self, images, labels):
        return self.forward(images, labels)
    
    def forward(self, images, labels):
        images = images.clone()
        labels = labels.clone()
        loss = torch.nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        
        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs, _ = self.model(adv_images)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images.clone()