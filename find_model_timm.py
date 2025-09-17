

import timm

model_list = timm.list_models('convnext*_laion*', pretrained=True)
model_list = timm.list_models('convnext*', pretrained=True)
model_list = timm.list_models('convnextv2_*fcmae*', pretrained=True)
for m in model_list:
    print(m)

model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=True , num_classes=2)

for name , parm in model.named_parameters():
    parm.requires_grad = False
    print(name, parm.requires_grad)

# for name , parm in model.named_parameters():
#     if name.startswith('head') or name.startswith('stages.3.blocks.0'):
#         parm.requires_grad = True

# for  name, param in model.named_parameters():
#     if parm.requires_grad:
#         print(name, param.requires_grad)


total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,} / {total:,} ({100.0*trainable/total:.2f}%)")


# model  = timm.create_model('convnext_base.clip_laion2b_augreg_ft_in1k', pretrained=True , num_classes=2)

# import torchinfo
# print(model)
# torchinfo.summary(model, input_size=(1, 3, 320 , 320))


# efficient_list = timm.list_models('tf_efficientnetv2*.in21k_ft_in1k*', pretrained=True)


# for m in efficient_list:

#     print(m)

    

