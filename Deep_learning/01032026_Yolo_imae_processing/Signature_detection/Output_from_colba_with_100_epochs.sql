Ultralytics 8.4.19 🚀 Python-3.12.12 torch-2.10.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=signature.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo26n.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=train2, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/runs/detect/train2, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5, 3, True]        
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    119808  ultralytics.nn.modules.block.C3k2            [384, 128, 1, True]           
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     34304  ultralytics.nn.modules.block.C3k2            [256, 64, 1, True]            
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     95232  ultralytics.nn.modules.block.C3k2            [192, 128, 1, True]           
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    463104  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True, 0.5, True]
 23        [16, 19, 22]  1    241566  ultralytics.nn.modules.head.Detect           [1, 1, True, [64, 128, 256]]  
YOLO26n summary: 260 layers, 2,504,190 parameters, 2,504,190 gradients, 5.8 GFLOPs

Transferred 606/708 items from pretrained weights
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1376.5±279.7 MB/s, size: 70.5 KB)
train: Scanning /content/datasets/signature/labels/train.cache... 143 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 143/143 37.5Mit/s 0.0s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 894.1±600.9 MB/s, size: 70.7 KB)
val: Scanning /content/datasets/signature/labels/val.cache... 35 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 35/35 1.5Mit/s 0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 114 weight(decay=0.0), 126 weight(decay=0.0005), 126 bias(decay=0.0)
Plotting labels to /content/runs/detect/train2/labels.jpg... 
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to /content/runs/detect/train2
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      2.42G     0.8268      5.986   0.008042         31        640: 100% ━━━━━━━━━━━━ 9/9 1.9it/s 4.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.3it/s 0.5s
                   all         35         35    0.00276      0.829      0.428      0.424

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      2.42G     0.7418      6.125   0.007331         23        640: 100% ━━━━━━━━━━━━ 9/9 2.9it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.6it/s 0.3s
                   all         35         35    0.00333          1      0.657      0.623

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      2.42G     0.6435      4.885   0.006332         37        640: 100% ━━━━━━━━━━━━ 9/9 3.6it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         35         35    0.00333          1      0.732      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      2.42G     0.7544      4.877   0.007768         32        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.1it/s 0.3s
                   all         35         35    0.00333          1      0.725       0.61

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      2.42G     0.7302      4.606   0.007723         21        640: 100% ━━━━━━━━━━━━ 9/9 2.9it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.8it/s 0.4s
                   all         35         35    0.00333          1      0.742       0.67

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      2.42G     0.8209      4.137   0.008355         34        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.9it/s 0.3s
                   all         35         35    0.00333          1      0.855       0.74

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      2.42G     0.8181      3.825   0.008262         31        640: 100% ━━━━━━━━━━━━ 9/9 3.6it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         35         35      0.652      0.215      0.766      0.687

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      2.42G     0.7942      3.609   0.008121         29        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.9it/s 0.3s
                   all         35         35      0.587      0.714       0.79      0.719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100      2.42G     0.8216      3.278   0.008449         38        640: 100% ━━━━━━━━━━━━ 9/9 2.5it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.3it/s 0.3s
                   all         35         35       0.98      0.686      0.865      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      2.42G       0.85      3.418   0.008801         33        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.1it/s 0.3s
                   all         35         35      0.961       0.71      0.948      0.873

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100      2.42G     0.8536       2.93   0.008247         38        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.2it/s 0.3s
                   all         35         35      0.939      0.857      0.976      0.863

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      2.42G     0.8485      2.851   0.008163         33        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.7it/s 0.3s
                   all         35         35      0.885        0.8      0.959      0.885

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100      2.42G     0.7445      2.763   0.007108         36        640: 100% ━━━━━━━━━━━━ 9/9 2.8it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.8it/s 0.3s
                   all         35         35      0.992      0.714       0.91      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100      2.42G     0.8028      2.501   0.007491         35        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.3it/s 0.3s
                   all         35         35          1      0.737       0.88      0.804

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100      2.42G     0.7281      2.292   0.006822         34        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.4it/s 0.3s
                   all         35         35       0.98      0.743      0.909      0.846

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100      2.42G      0.759       2.23   0.007632         39        640: 100% ━━━━━━━━━━━━ 9/9 3.1it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.3it/s 0.5s
                   all         35         35      0.944      0.958      0.982      0.897

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100      2.42G     0.7771      2.178   0.007647         33        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.3it/s 0.3s
                   all         35         35      0.953      0.886      0.986      0.912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100      2.42G     0.7472      2.103   0.007122         27        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.4it/s 0.3s
                   all         35         35          1      0.928      0.987      0.903

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100      2.42G     0.6976      1.842   0.006864         31        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.2it/s 0.3s
                   all         35         35          1      0.993      0.995      0.935

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100      2.42G     0.7755      1.726   0.007372         27        640: 100% ━━━━━━━━━━━━ 9/9 2.5it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.1it/s 0.3s
                   all         35         35          1      0.921      0.993       0.88

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100      2.42G     0.7564      1.684   0.007394         30        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.6it/s 0.4s
                   all         35         35      0.995      0.743      0.931      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100      2.42G     0.7308      1.642   0.006966         38        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.8it/s 0.3s
                   all         35         35      0.997        0.8      0.953      0.822

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100      2.42G     0.6728      1.426    0.00618         27        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.0it/s 0.3s
                   all         35         35       0.96          1      0.994      0.926

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100      2.42G     0.6448      1.413   0.006373         28        640: 100% ━━━━━━━━━━━━ 9/9 2.6it/s 3.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.9it/s 0.3s
                   all         35         35      0.943      0.942      0.979      0.912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100      2.42G      0.625      1.305   0.006289         33        640: 100% ━━━━━━━━━━━━ 9/9 3.6it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.9it/s 0.3s
                   all         35         35      0.893      0.958      0.968      0.925

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      2.42G      0.628      1.303   0.006188         27        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.0it/s 0.3s
                   all         35         35      0.919      0.974      0.978      0.924

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100      2.42G     0.6968      1.168   0.006789         34        640: 100% ━━━━━━━━━━━━ 9/9 3.2it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.9it/s 0.5s
                   all         35         35      0.895      0.975      0.981       0.92

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100      2.42G     0.6873      1.173    0.00741         34        640: 100% ━━━━━━━━━━━━ 9/9 3.2it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.6it/s 0.3s
                   all         35         35          1       0.91      0.989      0.925

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      2.42G     0.7412      1.111   0.007208         30        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.8it/s 0.3s
                   all         35         35          1      0.937      0.991      0.918

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100      2.42G     0.5953      1.032   0.005676         33        640: 100% ━━━━━━━━━━━━ 9/9 3.6it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.4it/s 0.3s
                   all         35         35       0.97      0.931      0.989      0.936

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100      2.42G     0.6818       1.03   0.006316         32        640: 100% ━━━━━━━━━━━━ 9/9 2.7it/s 3.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.9it/s 0.4s
                   all         35         35      0.971      0.958       0.99      0.928

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      2.42G     0.5966      1.011   0.005647         32        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.4it/s 0.3s
                   all         35         35      0.971       0.97       0.99      0.894

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100      2.42G     0.6424     0.8905   0.006151         30        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.9it/s 0.3s
                   all         35         35      0.966          1      0.992      0.913

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100      2.42G      0.605     0.8565   0.005842         24        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.1it/s 0.3s
                   all         35         35          1      0.961      0.993      0.951

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100      2.42G     0.5952     0.9066   0.006354         31        640: 100% ━━━━━━━━━━━━ 9/9 2.6it/s 3.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.9it/s 0.3s
                   all         35         35      0.946      0.971       0.99      0.908

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100      2.42G       0.56     0.8386   0.005701         35        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.4it/s 0.3s
                   all         35         35      0.995      0.971      0.993      0.943

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100      2.42G     0.5724      0.799   0.005468         40        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.2it/s 0.3s
                   all         35         35      0.985      0.971      0.993      0.929

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100      2.42G     0.6262      0.771   0.006495         37        640: 100% ━━━━━━━━━━━━ 9/9 3.2it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.9it/s 0.5s
                   all         35         35       0.92      0.989      0.988      0.883

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100      2.42G     0.6432     0.7807   0.006424         33        640: 100% ━━━━━━━━━━━━ 9/9 3.5it/s 2.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.0it/s 0.3s
                   all         35         35       0.85      0.972      0.973       0.85

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100      2.42G     0.6596     0.7671   0.006621         36        640: 100% ━━━━━━━━━━━━ 9/9 3.6it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.7it/s 0.3s
                   all         35         35      0.935      0.829      0.953      0.911

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100      2.42G     0.5682     0.7048   0.005515         30        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.8it/s 0.3s
                   all         35         35      0.943      0.953      0.986      0.924

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100      2.42G     0.5624     0.7635   0.005776         28        640: 100% ━━━━━━━━━━━━ 9/9 2.6it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.9it/s 0.4s
                   all         35         35          1      0.966      0.993      0.951

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100      2.42G      0.584        0.7   0.005157         39        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.9it/s 0.3s
                   all         35         35      0.937          1      0.993      0.889

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100      2.42G     0.5884     0.7201   0.005573         28        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         35         35      0.998          1      0.995      0.943

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100      2.42G     0.5606      0.624   0.005636         35        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.0it/s 0.3s
                   all         35         35      0.998          1      0.995       0.91

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100      2.42G     0.5611     0.5905   0.005292         33        640: 100% ━━━━━━━━━━━━ 9/9 2.6it/s 3.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.4it/s 0.3s
                   all         35         35          1      0.969      0.994      0.886

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100      2.42G     0.5796     0.6782    0.00587         33        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.2it/s 0.3s
                   all         35         35          1      0.969      0.994      0.953

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100      2.42G     0.5734     0.6479   0.005362         31        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.9it/s 0.3s
                   all         35         35      0.989          1      0.995      0.962

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100      2.42G     0.5505     0.5915   0.004951         37        640: 100% ━━━━━━━━━━━━ 9/9 3.4it/s 2.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.2it/s 0.5s
                   all         35         35      0.968          1      0.994      0.958

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100      2.42G     0.4877     0.5589   0.004844         32        640: 100% ━━━━━━━━━━━━ 9/9 3.4it/s 2.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.9it/s 0.3s
                   all         35         35      0.995          1      0.995      0.962

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100      2.42G     0.4152     0.5389   0.004087         26        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.9it/s 0.3s
                   all         35         35      0.994          1      0.995      0.975

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100      2.42G     0.4979     0.5592   0.004614         33        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.4it/s 0.3s
                   all         35         35          1      0.997      0.995      0.965

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100      2.42G       0.42     0.4788   0.004021         32        640: 100% ━━━━━━━━━━━━ 9/9 3.0it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.9it/s 0.5s
                   all         35         35          1      0.995      0.995      0.972

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100      2.42G     0.5005     0.5203    0.00518         35        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.8it/s 0.3s
                   all         35         35          1      0.998      0.995      0.969

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100      2.42G     0.4459     0.5319   0.004303         22        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.8it/s 0.3s
                   all         35         35          1      0.996      0.995      0.969

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100      2.42G     0.5187     0.5281   0.005186         42        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.0it/s 0.3s
                   all         35         35       0.89          1      0.985       0.94

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100      2.42G     0.4576     0.4813   0.004829         29        640: 100% ━━━━━━━━━━━━ 9/9 2.6it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.5it/s 0.3s
                   all         35         35      0.999          1      0.995       0.94

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100      2.42G     0.4772     0.4867   0.004485         32        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.6it/s 0.3s
                   all         35         35      0.999          1      0.995      0.917

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100      2.42G     0.3979     0.4689   0.003652         43        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.3it/s 0.3s
                   all         35         35          1          1      0.995      0.969

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100      2.42G      0.419     0.4651    0.00386         33        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.6it/s 0.3s
                   all         35         35          1      0.995      0.995      0.976

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100      2.42G      0.424     0.4568   0.004117         35        640: 100% ━━━━━━━━━━━━ 9/9 2.7it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.2it/s 0.3s
                   all         35         35      0.945      0.984      0.987      0.964

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100      2.42G      0.413     0.4316   0.003914         31        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.8it/s 0.3s
                   all         35         35      0.992      0.971      0.993       0.94

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100      2.42G     0.4269     0.4303   0.003947         28        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.6it/s 0.4s
                   all         35         35      0.889      0.915      0.975      0.925

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100      2.42G     0.4292     0.4427   0.003673         39        640: 100% ━━━━━━━━━━━━ 9/9 3.4it/s 2.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.6it/s 0.4s
                   all         35         35          1      0.943      0.988      0.942

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100      2.42G     0.4149      0.427   0.004017         29        640: 100% ━━━━━━━━━━━━ 9/9 3.2it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.5it/s 0.3s
                   all         35         35      0.968      0.971      0.988      0.947

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100      2.42G     0.4256     0.4655   0.004093         36        640: 100% ━━━━━━━━━━━━ 9/9 3.6it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         35         35      0.989      0.971      0.991      0.956

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100      2.42G     0.3899     0.3988   0.003793         27        640: 100% ━━━━━━━━━━━━ 9/9 4.0it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.3it/s 0.4s
                   all         35         35      0.986      0.971       0.99       0.96

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100      2.42G     0.3943     0.4135   0.003746         36        640: 100% ━━━━━━━━━━━━ 9/9 3.0it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.0it/s 0.5s
                   all         35         35      0.935      0.971      0.983      0.954

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100      2.42G      0.437     0.4256   0.004012         27        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.1it/s 0.3s
                   all         35         35      0.971      0.959      0.992      0.965

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100      2.42G     0.3529     0.3618   0.003356         32        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.0it/s 0.3s
                   all         35         35          1       0.94      0.993      0.964

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100      2.42G     0.3194     0.3661   0.003102         21        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.4it/s 0.3s
                   all         35         35      0.971      0.954      0.991       0.96

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100      2.42G     0.3935     0.3991   0.003628         34        640: 100% ━━━━━━━━━━━━ 9/9 2.5it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.4it/s 0.3s
                   all         35         35      0.917      0.946      0.989      0.963

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100      2.42G     0.3839     0.4321   0.003298         27        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.4it/s 0.3s
                   all         35         35      0.944      0.971      0.977       0.94

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100      2.42G     0.3882     0.4169   0.003396         34        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.5it/s 0.3s
                   all         35         35      0.919      0.971      0.993      0.955

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100      2.42G     0.3853     0.3759   0.003585         40        640: 100% ━━━━━━━━━━━━ 9/9 3.6it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.7it/s 0.3s
                   all         35         35      0.939          1      0.994      0.958

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100      2.42G     0.3901     0.3908   0.003711         29        640: 100% ━━━━━━━━━━━━ 9/9 2.6it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         35         35      0.979      0.943      0.989      0.962

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100      2.42G     0.3569     0.3518   0.003046         30        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.7it/s 0.3s
                   all         35         35      0.998      0.943      0.986      0.966

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100      2.42G     0.3775     0.3622   0.003341         42        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.3it/s 0.3s
                   all         35         35       0.98      0.943      0.987      0.964

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100      2.42G     0.3475     0.3864   0.003094         40        640: 100% ━━━━━━━━━━━━ 9/9 3.3it/s 2.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.3it/s 0.6s
                   all         35         35          1      0.934      0.987      0.968

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100      2.42G     0.3464     0.3349   0.003089         25        640: 100% ━━━━━━━━━━━━ 9/9 3.3it/s 2.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         35         35      0.992      0.943      0.981      0.965

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100      2.42G     0.3594     0.3817    0.00321         32        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.4it/s 0.3s
                   all         35         35       0.99      0.943      0.981      0.959

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100      2.42G      0.335     0.3156   0.003023         27        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.3it/s 0.4s
                   all         35         35      0.982      0.886       0.98      0.959

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100      2.42G     0.3442     0.3152   0.003102         34        640: 100% ━━━━━━━━━━━━ 9/9 2.7it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.8it/s 0.5s
                   all         35         35      0.928      0.943      0.977       0.96

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100      2.42G     0.3284     0.3642   0.003044         27        640: 100% ━━━━━━━━━━━━ 9/9 3.6it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.2it/s 0.3s
                   all         35         35      0.994      0.943      0.985      0.964

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100      2.42G     0.3321     0.3503    0.00302         33        640: 100% ━━━━━━━━━━━━ 9/9 3.6it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.0it/s 0.3s
                   all         35         35      0.996      0.943      0.985      0.965

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100      2.42G     0.3482     0.3394    0.00294         36        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.4it/s 0.4s
                   all         35         35          1      0.934      0.986      0.961

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100      2.42G     0.3461     0.3746   0.002865         29        640: 100% ━━━━━━━━━━━━ 9/9 2.6it/s 3.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.5it/s 0.3s
                   all         35         35      0.982      0.914      0.987      0.963

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100      2.42G     0.3247     0.3267   0.002846         29        640: 100% ━━━━━━━━━━━━ 9/9 3.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.9it/s 0.3s
                   all         35         35      0.998      0.943      0.991      0.968

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100      2.42G     0.3585     0.3466   0.003348         33        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.4it/s 0.3s
                   all         35         35       0.94          1      0.992      0.975

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100      2.42G     0.3367     0.3451   0.002857         32        640: 100% ━━━━━━━━━━━━ 9/9 3.5it/s 2.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.4it/s 0.6s
                   all         35         35      0.945      0.974      0.992      0.976
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100      2.42G     0.2671     0.3273   0.002762         15        640: 100% ━━━━━━━━━━━━ 9/9 1.9it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.6it/s 0.4s
                   all         35         35      0.991          1      0.995      0.973

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100      2.42G     0.2292     0.3125   0.002423         15        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.3it/s 0.3s
                   all         35         35      0.967          1      0.994      0.979

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100      2.42G     0.2298     0.3267   0.002672         15        640: 100% ━━━━━━━━━━━━ 9/9 3.0it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.2it/s 0.5s
                   all         35         35      0.972      0.983      0.994      0.981

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100      2.42G     0.2441     0.3082   0.002651         15        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.0it/s 0.3s
                   all         35         35      0.988      0.914      0.989      0.973

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100      2.42G     0.2144     0.2853   0.002282         15        640: 100% ━━━━━━━━━━━━ 9/9 4.0it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.6it/s 0.3s
                   all         35         35      0.987      0.914      0.987       0.97

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100      2.42G     0.2143     0.3049   0.002303         15        640: 100% ━━━━━━━━━━━━ 9/9 3.9it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.4it/s 0.3s
                   all         35         35      0.997      0.914      0.989      0.976

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100      2.42G      0.201     0.2759   0.002304         15        640: 100% ━━━━━━━━━━━━ 9/9 2.6it/s 3.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.1it/s 0.5s
                   all         35         35      0.914          1      0.989      0.979

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100      2.42G     0.1954     0.2885   0.002181         15        640: 100% ━━━━━━━━━━━━ 9/9 4.0it/s 2.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.0it/s 0.3s
                   all         35         35      0.915          1      0.991      0.974

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      2.42G     0.1961     0.2911   0.002002         15        640: 100% ━━━━━━━━━━━━ 9/9 4.0it/s 2.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.7it/s 0.3s
                   all         35         35      0.995          1      0.995      0.981

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      2.42G     0.1982     0.2716   0.002074         15        640: 100% ━━━━━━━━━━━━ 9/9 3.8it/s 2.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         35         35      0.993          1      0.995      0.987

100 epochs completed in 0.099 hours.
Optimizer stripped from /content/runs/detect/train2/weights/last.pt, 5.4MB
Optimizer stripped from /content/runs/detect/train2/weights/best.pt, 5.4MB

Validating /content/runs/detect/train2/weights/best.pt...
Ultralytics 8.4.19 🚀 Python-3.12.12 torch-2.10.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
YOLO26n summary (fused): 122 layers, 2,375,031 parameters, 0 gradients, 5.2 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.9it/s 0.3s
                   all         35         35      0.992          1      0.995      0.987
Speed: 0.1ms preprocess, 3.0ms inference, 0.0ms loss, 0.4ms postprocess per image
Results saved to /content/runs/detect/train2
