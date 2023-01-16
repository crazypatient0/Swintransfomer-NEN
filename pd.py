import time

import torch.utils.data.distributed
from PIL import Image
from torch.autograd import Variable
from models import build_model
from config import get_config
import argparse
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
try:
    from torchvision.transforms import InterpolationMode
    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp
import torch
import torchvision.transforms as transforms


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer Test script', add_help=False)
    parser.add_argument('--cfg', default='configs/swinv2/swinv2_tiny_patch4_window16_256.yaml', type=str, metavar="FILE",
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', default='output/swinv2_tiny_patch4_window16_256/default/ckpt_epoch_17.pth',
                        help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--local_rank", default='0', type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--ckp_path', help='tag of experiment')
    parser.add_argument('--img_path', help='tag of experiment')
    parser.add_argument('--model_acc', help='tag of experiment')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def model_ana():
    classes = ("Tumor1N0_Non-Metastatic", "Tumor1N1_Metastatic")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args, config = parse_option()
    model = build_model(config)
    ckp_path = args.ckp_path
    checkpoint = torch.load(ckp_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    model.to(DEVICE)
    def build_transform(is_train, config):
        # 判断 resize是否大于32
        resize_im = config.DATA.IMG_SIZE > 32
        if is_train:
            #生成transform 工具类
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,#输入大小
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,#亮度对比度饱和度色相偏移，详见config
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
            )
            # 如果图片过于小，就使用随机剪裁函数
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
            return transform

        t = []
        if resize_im:
            if config.TEST.CROP:
                size = int((256 / 224) * config.DATA.IMG_SIZE)
                t.append(
                    transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                    # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
            else:
                t.append(
                    transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                      interpolation=_pil_interp(config.DATA.INTERPOLATION))
                )
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)
    transform = build_transform(False, config)
    path = args.img_path
    total=0
    total +=1
    img = Image.open(path).convert('RGB')
    imgsize = img.size
    if imgsize[0]<512 or imgsize[1] <512:
        img = img.resize((512,512))
    img = transform(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    prob = (out.data.softmax(dim=-1)).cpu().numpy().tolist()[0]
    _, pred = torch.max(out.data, 1)
    predid = pred.data.cpu().numpy().tolist()[0]
    model_acc = args.model_acc
    pb = float(prob[predid])
    re_dict = {}
    re_dict['Prediction'] = classes[pred.data.item()]
    re_dict['Possibility'] = round(pb,3)
    print(re_dict)


if __name__ == '__main__':
    model_ana()