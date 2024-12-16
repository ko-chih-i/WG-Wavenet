class hparams:
    seed = 0
    ################################
    # Audio                        #
    ################################
    num_mels = 80
    num_freq = 513
    sample_rate = 22050
    frame_shift = 200
    frame_length = 800
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20
    fmin = 0
    fmax = 8000
    seg_l = 16000

    ################################
    # Train	                       #
    ################################
    is_cuda = True
    pin_mem = True
    n_workers = 4
    prep = False
    pth = None
    lr = 4e-4
    sch = False  # 不啟用學習率調度器
    max_iter = 1310  # 总迭代次数
    batch_size = 10  # 每个 batch 的样本数
    iters_per_log = 1310  # 每个 epoch 打印一次日志
    iters_per_sample = 1310  # 每个 epoch 保存一次样本
    iters_per_ckpt = 1310  # 每个 epoch 保存一次检查点

    gn = 10  
    n=5
    ################################
    # Model                        #
    ################################
    up_scale = [2, 5, 2, 5, 2]  # assert product = frame_shift
    sigma = 0.6
    n_flows = 4
    n_group = 8
    # for WN
    n_layers = 7
    n_channels = 128
    kernel_size = 3
    # for PF
    PF_n_layers = 7
    PF_n_channels = 64

    ################################
    # Spectral Loss                #
    ################################
    mag = True
    mel = True
    fft_sizes = [2048, 1024, 512, 256, 128]
    hop_sizes = [400, 200, 100, 50, 25]
    win_lengths = [2000, 1000, 500, 200, 100]
    mel_scales = [4, 2, 1, 0.5, 0.25]
