import timm
import torch

class EfficientNetV2_S(torch.nn.Module):
    def __init__(self):
        super(EfficientNetV2_S, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True , num_classes=2)
        
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
        
        print("--- Checking requires_grad status inside __init__ ---")
        for name, param in self.model.named_parameters():
            # if name.startswith('blocks.5'):
            print(f"name: {name}, requires_grad: {param.requires_grad}")
        print("----------------------------------------------------")
    

    def forward(self, x):
        x = self.model(x)
        return x
    

class EfficientNetV2_L(torch.nn.Module):
    def __init__(self):
        super(EfficientNetV2_L, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_l.in21k_ft_in1k', pretrained=True , num_classes=2)
        
        for param in model.parameters():
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

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    model = EfficientNetV2_S()
    print(model)
    # print("========================================")

    # print("\n--- Final requires_grad status outside __init__ ---")
    # for name, param in model.named_parameters():
    #     print(f"name : {name} , requires_grad : {param.requires_grad}")


    import torchinfo

    torchinfo.summary(
        model=model,
        input_size=(1, 3, 320, 320),
        verbose=True,
        col_names=["input_size", "output_size", "trainable"],
        row_settings=["depth"],
        mode='eval'
    )

