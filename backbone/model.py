import timm
import torch
import torch.nn as nn
from torch.autograd import Function


def build_param_groups_lrd(model: nn.Module):
    """Build parameter groups with layer-wise decayed LRs for EfficientNetV2-S.

    Priority order: head > stage5 > stage4 > (BN affine global) > other
    """
    head, s5, s4, bn_affine, other = [], [], [], [], []
    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)

        if any(k in name for k in ["classifier", "conv_head"]) or name.startswith("bn2"):
            head.append(p)
            seen.add(pid)
            continue
        if name.startswith("blocks.5."):
            s5.append(p)
            seen.add(pid)
            continue

        if ("bn" in name) and (name.endswith("weight") or name.endswith("bias")) and pid not in seen:
            bn_affine.append(p)
            seen.add(pid)
            continue

        other.append(p)

    groups = []
    if head:
        groups.append({"params": head, "lr": 1e-3})
    if s5:
        groups.append({"params": s5, "lr": 3e-4})
    if s4:
        groups.append({"params": s4, "lr": 2e-4})
    if bn_affine:
        groups.append({"params": bn_affine, "lr": 1e-4})
    if other:
        groups.append({"params": other, "lr": 1e-4})

    return groups



def set_trainable_efficientnet_v2s(model: nn.Module, unfreeze_stages, train_bn_affine: bool = True, verbose: bool = True):
      
    for _, p in model.named_parameters():
        p.requires_grad = False

    train_prefixes = ["classifier", "conv_head", "bn2"]
    for s in unfreeze_stages:
        train_prefixes.append(f"blocks.{s}")

    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in train_prefixes):
            p.requires_grad = True

    if train_bn_affine:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.requires_grad = True

    if verbose:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,} / {total:,} ({100.0*trainable/total:.2f}%)")

    return model



class EfficientNetV2_S_improved(torch.nn.Module):
    def __init__(self , num_classes = 2):
        # Fix incorrect super class reference
        super(EfficientNetV2_S_improved, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True , num_classes=num_classes)
        set_trainable_efficientnet_v2s(self.model, unfreeze_stages=("5.14","5.13"), train_bn_affine=True, verbose=True)
    

    def forward(self, x, return_features: bool = False):
        features = self.model.forward_features(x)
        logits = self.model.forward_head(features, pre_logits=False)
        if return_features:
            return logits, features
        return logits
    

class EfficientNetV2_S(torch.nn.Module):
    def __init__(self , num_classes = 2):
        super(EfficientNetV2_S, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True , num_classes=num_classes)
        
        # print("--- All parameter names inside __init__ ---")
        # for name, param in self.model.named_parameters():
        #     print(name)
        # print("-------------------------------------------")


        # for name , param in self.model.named_parameters():
        #     if name.startswith('classifier') or name.startswith('conv_head') or name.startswith('bn2'):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False


        for name , param in self.model.named_parameters():
            if name.startswith('classifier') or name.startswith('conv_head') or name.startswith('bn2') or name.startswith('blocks.5.14') :
                param.requires_grad = True
            else:
                param.requires_grad = False
        

        # for name , param in self.model.named_parameters():
        #     if name.startswith('classifier') or name.startswith('conv_head') or name.startswith('bn2') or name.startswith('blocks.5.14') or name.startswith('blocks.5.13'):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        
        # print("--- Checking requires_grad status inside __init__ ---")
        # for name, param in self.model.named_parameters():
        #     # if name.startswith('blocks.5'):
        #     print(f"name: {name}, requires_grad: {param.requires_grad}")
        # print("----------------------------------------------------")
    

    def forward(self, x, return_features: bool = False):
        features = self.model.forward_features(x)
        logits = self.model.forward_head(features, pre_logits=False)
        if return_features:
            return logits, features
        return logits
    
class EfficientNetV2_L(torch.nn.Module):
    def __init__(self , num_classes = 2):
        super(EfficientNetV2_L, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_l.in21k_ft_in1k', pretrained=True , num_classes= num_classes)
        
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.blocks[6][6].parameters() :
            param.requires_grad = True

        for parm in self.model.blocks[6][5].parameters() :
            parm.requires_grad = True

        print("Unfreezing parameters in self.model.classifier...")
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        for parm in self.model.conv_head.parameters():
            parm.requires_grad = True

        for parm in self.model.bn2.parameters():
            parm.requires_grad = True



        print("Verifying parameter freeze status:")
        total_params = 0
        trainable_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"- {name} (Trainable)")


        
        # print("--- Checking requires_grad status inside __init__ ---")
        # for name, param in self.model.named_parameters():
        #     print(f"name: {name}, requires_grad: {param.requires_grad}")
        # print("----------------------------------------------------")


    def forward(self, x, return_features: bool = False):
        features = self.model.forward_features(x)
        logits = self.model.forward_head(features, pre_logits=False)
        if return_features:
            return logits, features
        return logits

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class EfficientNetV2_S_DANN(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN) implementation for EfficientNetV2-S.
    """
    def __init__(self, num_classes=2, num_domains=2):
        super(EfficientNetV2_S_DANN, self).__init__()
        
        
        # --- Feature Extractor ---
        self.feature_extractor = timm.create_model(
            'tf_efficientnetv2_s.in21k', 
            pretrained=True, 
            num_classes=0  # Setting num_classes=0 returns the feature extractor
        )
        num_features = self.feature_extractor.num_features

        # --- Label Predictor ---
        self.label_predictor = nn.Linear(num_features, num_classes)

        # Align trainable policy with EfficientNetV2-S: head + blocks.5 + blocks.4 + BN affine
        set_trainable_efficientnet_v2s(self.feature_extractor, unfreeze_stages=("5.14",), train_bn_affine=True, verbose=False)
        
        # --- Domain Classifier ---
        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_domains)
        )
        
        self.grl = GradientReversalFunction.apply

        for name , param in self.domain_classifier.named_parameters():
            param.requires_grad = True

    def forward(self, x, alpha=1.0):
        """
        Forward pass for the DANN model.
        
        Args:
            x (torch.Tensor): Input tensor.
            alpha (float): The hyperparameter to trade off the domain loss.
        
        Returns:
            tuple: A tuple containing (label_output, domain_output).
        """
        features = self.feature_extractor(x)
        label_output = self.label_predictor(features)
        
        reversed_features = self.grl(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        
        return label_output, domain_output
    

# class GradientReversalFunction(Function):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.alpha
#         return output, None

# class GradientReversalLayer(nn.Module):
#     def __init__(self, alpha=1.0):
#         super(GradientReversalLayer, self).__init__()
#         self.alpha = alpha

#     def forward(self, x):
#         return GradientReversalFunction.apply(x, self.alpha)


class ConvNext_V2_Tiny(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNext_V2_Tiny, self).__init__()
        self.model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=True , num_classes=num_classes)
        
        for name , param in self.model.named_parameters():
            param.requires_grad = False

        for name , param in self.model.named_parameters():
            if name.startswith('head') or name.startswith('stages.3.blocks.0'):
                param.requires_grad = True

        print("Verifying parameter freeze status:")
        total_params = 0
        trainable_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"- {name} (Trainable)")

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Percentage of trainable parameters: {100.0 * trainable_params / total_params:.2f}%")

    def forward(self, x, return_features: bool = False):
        features = self.model.forward_features(x)
        logits = self.model.forward_head(features, pre_logits=False)
        if return_features:
            return logits, features
        return logits


if __name__ == "__main__":

    model = ConvNext_V2_Tiny()
    


    # model = EfficientNetV2_S_improved()

    # backbone_params = sum(p.numel() for p in model.parameters())
    # trainable_backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Model total params: {backbone_params:,}")
    # print(f"Model trainable params: {trainable_backbone_params:,}")
    # print(f"traineable params percentage: {100 * trainable_backbone_params / backbone_params:.2f}%")
    # print('==' * 30)

