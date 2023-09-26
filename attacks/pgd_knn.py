import torch
from utils import knn_score


class KNNPGDAttack:
    """PGD attack optimization on knn_score"""
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
    
    def __call__(self, images, feature_space):
        return self.forward(images, feature_space)
    
    def forward(self, images, feature_space):
        images = images.clone()
        feature_space = feature_space.clone()
        delta = torch.zeros_like(images, requires_grad=True)

        for _ in range(self.steps):
            adv_inputs = images + delta
            outputs, features = self.model(adv_inputs)

            # Compute the knn_score using the perturbed features
            distances = knn_score(feature_space, features.detach().cpu().numpy())

            # Compute the loss as the negative knn_score
            loss = -torch.from_numpy(distances).to(adv_inputs.device)

            loss.backward()

            # Update delta with the gradient
            delta.data = (delta + self.alpha * delta.grad.detach().sign()).clamp(-self.eps, self.eps)
            delta.grad.zero_()

        # Create the adversarial examples
        adv_inputs = images + delta.detach()
        adv_inputs = torch.clamp(adv_inputs, 0, 1)

        return adv_inputs