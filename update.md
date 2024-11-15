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

10 月 22 日下午 by lingxiaoyu (finetune框架debug完成，并开始finetune)
- debug完finetune.py文件，开始DiffVC下的模型finetune
- 存在一个问题：nn.linear中的weight会被放到cpu上，并且nn.linear.weight没有requires_grad，这个问题仍待解决(现在暂时的方案为：不更新nn.linear的weight，转而更新nn.linear.bias)

10 月 26 日 by lingxiaoyu (finetune的效果不明显，效果不佳，可能是一个decoder同时提取鲁棒和脆弱水印产生了互相影响，因此构建两个decoder分别提取鲁棒和脆弱水印)
- 构建multi_decoder_half_vulnerable.py、configs/multi_decoder_train.yaml

10 月 30 日 by lingxiaoyu 
- 构建multi_decoder_finetune.py，根据multi_decoder的保存路径开展finetune(finetune 几个epoch并检查效果)
- 构建multi_decoder_one_watermark_finetune.py, 使用两个decoder提取同一个水印

11 月 8 日 by yuFeng
- 添加了官方实现的 AutoVC ，但是这个版本的 VC 效果不太好，未上传的模型参数文件在 `./deepFake/autovc/pretrained_models/*` 下

11 月 11 日 by yuFeng
- 添加了 [VALL-E-X](https://github.com/Plachtaa/VALL-E-X)，已将模型参数添加到 gitignore，理论上第一次运行时会自己下载，不过这个好像有点慢，不知道是不是现在 GPU 太多人用的问题qwq

11 月 11 日 by lingxiaoyu
- 修改了My_model/modules.py Decoder dl中的内容。将distrotion的作用方式进行了修改，这样就可以不用考虑dl过程中如何保持音频梯度仍然可回传的问题，可以让音频在YourTTS、AutoVC等情况下正常运行。