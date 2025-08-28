import timm
import torch
# model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True , num_classes=2)
# model = model.eval()

# import torchinfo

# torchinfo.summary(model, input_size=(1, 3, 320 , 320))


class EfficientNetV2_S(torch.nn.Module):
    def __init__(self):
        super(EfficientNetV2_S, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True , num_classes=2)
        
        for name , param in self.model.named_parameters():
            if name.startswith('classifier') or name.startswith('conv_head') or name.startswith('bn2'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
            print(f"name : {name} , requires_grad : {param.requires_grad}")
        
    def forward(self, x):
        x = self.model(x)
        return x
    

# if __name__ == "__main__":
#     model = EfficientNetV2_S(num_classes=2)
#     print(model)
#     import torchinfo
#     model_info = torchinfo.summary(
#     model=model,
#     input_size=(1, 3, 320, 320),
#     verbose=False,
#     col_names=["input_size", "output_size","trainable"],
#     row_settings=["depth"],
#     device='cuda:1',
#     mode='eval'
# )
#     print(model_info)

    # for name,ch in model.named_children():
    #     print("name :",name)
    #     print("child :", ch)
    #     print('===========================')