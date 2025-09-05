import timm
import torch
import torch.nn as nn
from torch.autograd import Function

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
    

    def forward(self, x):
        x = self.model(x)
        return x
    

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


    def forward(self, x):
        x = self.model(x)
        return x

# --- Domain-Adversarial Training Components ---

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

"""
adversarial domain adaptation model
"""
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

        for name , param in self.feature_extractor.named_parameters():
            if name.startswith('classifier') or name.startswith('conv_head') or name.startswith('bn2') or name.startswith('blocks.5.14') :
                param.requires_grad = True
            else:
                param.requires_grad = False
        
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



if __name__ == "__main__":
    # model = EfficientNetV2_S()
    # print(model)
    # print("========================================")

    # print("\n--- Final requires_grad status outside __init__ ---")
    # for name, param in model.named_parameters():
    #     print(f"name : {name} , requires_grad : {param.requires_grad}")


    model = EfficientNetV2_L()
    #print(model)
    import torchinfo

    torchinfo.summary(
        model=model,
        input_size=(1, 3, 640, 640),
        verbose=True,
        col_names=["input_size", "output_size", "trainable"],
        row_settings=["depth"],
        mode='eval'
    )

    backbone_params = sum(p.numel() for p in model.parameters())
    trainable_backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model total params: {backbone_params:,}")
    print(f"Model trainable params: {trainable_backbone_params:,}")
    print(f"traineable params percentage: {100 * trainable_backbone_params / backbone_params:.2f}%")
    print('==' * 30)