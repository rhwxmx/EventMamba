## [Rethinking Efficient and Effective Point-based Networks for Event Camera Classification and Regression: EventMamba](https://arxiv.org/abs/2405.06116)

<img src="figures/eventmamba.png" alt="EventMamba's architecture" width="800" />

### Installation

    conda create -n eventmamba python=3.8
    conda activate eventmamba
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install h5py,tqdm,scikit-learn,tensorboard
    pip install spikingjelly
    pip install mamba-ssm
    install the cuda kernel: https://github.com/erikwijmans/Pointnet2_PyTorch
    
### Usage
1. Prepare the data:

        cd dataprocess
        python generate_xxx.py

2. Put the train.h5 and test.h5 to ./data/xxx/:

3. Modify the num_class, data_path, log_name and others.

4. Run the train script:
    
    For action recognition:
        
        python train_classification.py
    For camera pose relocalization:

        python train_odometry.py
    For eye tracking task:

        python train_eye_tracking.py

### Configuration

| Dataset              | Version | Dimension     | Group            | Accuracy |
| -------------------- | ------- | ------------- | ---------------- | -------- |
| DVSGesture           | V2      | [16, 32, 64]  | [512, 256, 128]  | 0.992    |
| DVSGesture           | V2      | [24, 48, 96]  | [512, 256, 128]  | 0.996    |
| DailyDVS             | V2      | [32, 64, 128] | [1024, 512, 256] | 0.993    |
| DVSAction            | V2      | [32, 64, 128] | [512, 256, 128]  | 0.893    |
| HMDB51-DVS (filename) | V1      | [64, 128, 256] | [1024, 512, 256] | 0.604    |
| HMDB51-DVS (sliding)  | V1      | [64, 128, 256] | [1024, 512, 256] | 0.864    |
| UCF101-DVS (filename) | V2      | [32, 64, 128] | [1024, 512, 256] | 0.903    |
| UCF101-DVS (sliding)  | V1      | [64, 128, 256] | [1024, 512, 256] | 0.979    |
| THU-CHA              | V1      | [64, 128, 256] | [1024, 512, 256] | 0.594    |
| IJRR                 | V1      | [32, 64, 128] | [512, 256, 128]  | -        |
| 3ET(+sigmoid)        | V1      | [32, 64, 128] | [512, 256, 128] | 0.951    |

### Download

| Dataset    | Split    | Data, log and pretrained model |
| ---------- | -------- | -------- |
| DVSGesture | offical  |   [DOWNLOAD](https://pan.baidu.com/s/1uNM4tc3WHwDIB8BaR-Ygcg?pwd=GEST) extract code: GEST |
| DailyDVS   | filename | [DOWNLOAD](https://pan.baidu.com/s/1zDERP3MBhk9XL6jcfn1Ggw?pwd=DAIL) extract code: DAIL     |
| DVSAction  | offical  | [DOWNLOAD](https://pan.baidu.com/s/1EzeXI5xb9OlAnw-QpSYZ9g?pwd=ACTI) extract code: ACTI     |
| HMDB51-DVS     | filename/sliding window     | [DOWNLOAD](https://pan.baidu.com/s/1w7SYrAVDrEK0t-Q4CulaAw?pwd=HMDB) extract code:HMDB |
| UCF101-DVS     | filename/sliding window    | [DOWNLOAD](https://pan.baidu.com/s/11admC1576VM0Vt7tZTSbJw?pwd=UCF1) extract code:UCF1     |
| THU-CHL    | offical     | [DOWNLOAD](https://pan.baidu.com/s/120urKUfRJu_xHM6nCnNOyA?pwd=THUC)  extract code:THUC   |
| IJRR       |   sliding window   | [DOWNLOAD](https://pan.baidu.com/s/1AxzvFYAD9dLnYrMIpwkuag?pwd=IJRR) extract code: IJRR  |
| 3ET        | offical     | [DOWNLOAD](https://pan.baidu.com/s/1XPvl__K-R-B1uTlOhPaeeA?pwd=3ETE) extract code: 3ETE  |

if you want use the pretrained model, please use the (dists = square_distance(new_xyz, xyz)) for version 1 and use the (dists = square_distance(new_points, points)) for version 2 in module.py.

### Citation
If you find our work useful in your research, please consider citing:

    @article{ren2024rethinking,
    title={Rethinking Efficient and Effective Point-based Networks for Event Camera Classification and Regression: EventMamba},
    author={Ren, Hongwei and Zhou, Yue and Zhu, Jiadong and Fu, Haotian and Huang, Yulong and Lin, Xiaopeng and Fang, Yuetong and Ma, Fei and Yu, Hao and Cheng, Bojun},
    journal={arXiv preprint arXiv:2405.06116},
    year={2024}
    }
and this paper is an expansion of our previous two works [TTPOINT](https://dl.acm.org/doi/abs/10.1145/3581783.3612258?casa_token=z72pohcxZTAAAAAA:pO42EmMVOEp-8PJPx4WBUwJyjrs-K2Z7lkWbZsanCTF72u763LuxdWNPYAXuTKUT4g82yPgPgLbLH6I) and [PEPNet](https://openaccess.thecvf.com/content/CVPR2024/html/Ren_A_Simple_and_Effective_Point-based_Network_for_Event_Camera_6-DOFs_CVPR_2024_paper.html):

    @inproceedings{ren2023ttpoint,
    title={Ttpoint: A tensorized point cloud network for lightweight action recognition with event cameras},
    author={Ren, Hongwei and Zhou, Yue and Fu, Haotian and Huang, Yulong and Xu, Renjing and Cheng, Bojun},
    booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
    pages={8026--8034},
    year={2023}
    }
    @inproceedings{ren2024simple,
    title={A Simple and Effective Point-based Network for Event Camera 6-DOFs Pose Relocalization},
    author={Ren, Hongwei and Zhu, Jiadong and Zhou, Yue and Fu, Haotian and Huang, Yulong and Cheng, Bojun},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={18112--18121},
    year={2024}
    }    

### Acknowledgment
Thanks to the previous works, PointNet, PointNet++, PointMLP and STNet.    