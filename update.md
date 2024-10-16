# 更新记录

10 月 12 日早上 by yuFeng (add VQMIVC and DiffVC):

- 修改了 dataset 和 hifigan 的 gitignore 规则，添加了 dataset 和 hifigan 的代码，保留了对数据集和模型参数本身的 gitignore
- 添加文件 `valid.py` 可以直接对训练好的模型进行验证测试，在命令行执行代码时，其加载的 `-t` 为 `-t config/valid.yaml`
- 删除了 `demo.py`，这个文件是当时给水印平台那边展示的时候用的，所以直接删了
- 将绝对路径修改为了相对路径（这里是我比较怕会出问题的更新）
- 现有的更新我还是直接添加到了 `dl.py` 里，这样更简单点，后期如果 `dl.py` 其它的 attack_type 不用的话再重构一下吧
- 添加了 Diff-VC，路径为 deepFake/DiffVC，未上传的模型参数文件可以参照 `.gitignore`，可以直接在我的目录（`/amax/home/zhaoxd/develop/Traceable_watermark`）下复制或跟着 [github](https://github.com/trinhtuanvubk/Diff-VC/tree/main) 说明下载
- 添加了 VQMIVC，路径为 deepFake/VQMIVC，未上传的模型参数文件同样可以在我的目录下复制或者跟着 [github](https://github.com/Wendison/VQMIVC) 的说明下载，**另外，需要根据他们 [github](https://github.com/Wendison/VQMIVC) 的说明配置 vocoder**
    - 但是这里有个问题，即预训练的 VQMIVC 的 sr 是 16000，而我们的添加水印后的音频的 sr 是 22050，这就导致在添加水印后的音频想要过 VQMIVC 就需要转换 sr，在过完 VQMIVC 之后再 resample 回去，这样的重采样过程是否会对水印造成损害？
    - 这个问题已经被解决，因为base模型可以在训练过程中添加resample扰动，使得水印具备抗resample的能力

10 月 12 日下午 by lingxiaoyu (冗余文件删除)
- 删除half_vunlerable_copy.pyhalf_vunlerable_copy.py是一个和half_vunlerable.py一样的文件，在loss计算中减少了fragile watermark loss，已经被废弃，因此删除
- 删除Traceable_train_0104.py, Traceable_train_0216_16bit copy.py, train_1220.py。

10 月 12 日下午 by lingxiaoyu (base模型重训)
- 修改train.yaml中ckpt文件保存路径
- 重新训练半易碎模型base，模型保存于./results/ckpt/half_vunlerable_watermark内

10 月 16 日下午 by lingxiaoyu (finetune框架搭建)
- 创建新的finetune.py文件，用于现有模型微调
- 增加的方法：使用SVD方法分割参数，把训练参数指定为对角矩阵，SVD方法详见./My_model/model_SVD.py