class DefaultConfig():
    #backbone
    pretrained = True
    # pretrained = False
    freeze_stage_1=True
    freeze_bn=True

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    class_num=20
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    # limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    # limit_range = [[-1, 0.125], [0.125, 0.25], [0.25, 0.5], [0.5, 1], [1, 999999]]
    # limit_range = [[-1, 0.0625], [0.0625, 0.125], [0.125, 0.25], [0.25, 0.5], [0.5, 2]]
    limit_range = [[-1, 32], [32, 64], [64, 128], [128, 256], [256, 999999]]

    #inference
    score_threshold = 0.05
    nms_iou_threshold = 0.5       # 0.75 0.5
    max_detection_boxes_num = 500
