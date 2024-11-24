import torch 
from timm.models import create_model
import models  # noqa: F401
from infer_for_test import *
from decord import VideoReader 

args, ds_init = get_args()

checkpoint = torch.load(args.finetune, map_location='cpu')
print("For inference, Load ckpt from %s" % args.finetune)

checkpoint_model = None
for model_key in args.model_key.split('|'):
    if model_key in checkpoint:
        checkpoint_model = checkpoint[model_key]
        print("Load state_dict by model_key = %s" % model_key)
        break
if checkpoint_model is None:
    checkpoint_model = checkpoint
for old_key in list(checkpoint_model.keys()):
    if old_key.startswith('_orig_mod.'):
        new_key = old_key[10:]
        checkpoint_model[new_key] = checkpoint_model.pop(old_key)
            
checkpoint = torch.load(args.finetune, map_location='cpu')
print("Load ckpt from %s" % args.finetune)

checkpoint_model = None
for model_key in args.model_key.split('|'):
    if model_key in checkpoint:
        checkpoint_model = checkpoint[model_key]
        print("Load state_dict by model_key = %s" % model_key)
        break
if checkpoint_model is None:
    checkpoint_model = checkpoint
for old_key in list(checkpoint_model.keys()):
    if old_key.startswith('_orig_mod.'):
        new_key = old_key[10:]
        checkpoint_model[new_key] = checkpoint_model.pop(old_key)

model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        head_drop_rate=args.head_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        with_cp=args.with_checkpoint,
    )
model.reset_classifier(10)

patch_size = model.patch_embed.patch_size

args.window_size = (args.num_frames // args.tubelet_size,
                    args.input_size // patch_size[0],
                    args.input_size // patch_size[1])

args.patch_size = patch_size

state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[
            k].shape != state_dict[k].shape:
        if checkpoint_model[k].shape[
                0] == 710 and args.data_set.startswith('Kinetics'):
            print(f'Convert K710 head to {args.data_set} head')
            if args.data_set == 'Kinetics-400':
                label_map_path = 'misc/label_710to400.json'
            elif args.data_set == 'Kinetics-600':
                label_map_path = 'misc/label_710to600.json'
            elif args.data_set == 'Kinetics-700':
                label_map_path = 'misc/label_710to700.json'
            elif args.data_set == 'Kinnetics-10' :
                label_map_path = 'misc/label_710to10.json'
            label_map = json.load(open(label_map_path))
            checkpoint_model[k] = checkpoint_model[k][label_map]
        else:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True
    
all_keys = list(checkpoint_model.keys())
new_dict = OrderedDict()
for key in all_keys:
    if key.startswith('backbone.'):
        new_dict[key[9:]] = checkpoint_model[key]
    elif key.startswith('encoder.'):
        new_dict[key[8:]] = checkpoint_model[key]
    else:
        new_dict[key] = checkpoint_model[key]
checkpoint_model = new_dict

# interpolate position embedding
if 'pos_embed' in checkpoint_model:
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
    num_patches = model.patch_embed.num_patches  #
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

    # height (== width) for the checkpoint position embedding
    orig_size = int(
        ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) //
            (args.num_frames // model.patch_embed.tubelet_size))**0.5)
    # height (== width) for the new position embedding
    new_size = int(
        (num_patches //
            (args.num_frames // model.patch_embed.tubelet_size))**0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" %
                (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # B, L, C -> BT, H, W, C -> BT, C, H, W
        pos_tokens = pos_tokens.reshape(
            -1, args.num_frames // model.patch_embed.tubelet_size,
            orig_size, orig_size, embedding_size)
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                        embedding_size).permute(
                                            0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens,
            size=(new_size, new_size),
            mode='bicubic',
            align_corners=False)
        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
            -1, args.num_frames // model.patch_embed.tubelet_size,
            new_size, new_size, embedding_size)
        pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
elif args.input_size != 224:
    pos_tokens = model.pos_embed
    org_num_frames = 16
    T = org_num_frames // args.tubelet_size
    P = int((pos_tokens.shape[1] // T)**0.5)
    C = pos_tokens.shape[2]
    new_P = args.input_size // patch_size[0]
    # B, L, C -> BT, H, W, C -> BT, C, H, W
    pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
    pos_tokens = pos_tokens.reshape(-1, P, P, C).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens,
        size=(new_P, new_P),
        mode='bicubic',
        align_corners=False)
    # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
    pos_tokens = pos_tokens.permute(0, 2, 3,
                                    1).reshape(-1, T, new_P, new_P, C)
    pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
    model.pos_embed = pos_tokens  # update
if args.num_frames != 16:
    org_num_frames = 16
    T = org_num_frames // args.tubelet_size
    pos_tokens = model.pos_embed
    new_T = args.num_frames // args.tubelet_size
    P = int((pos_tokens.shape[1] // T)**0.5)
    C = pos_tokens.shape[2]
    pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
    pos_tokens = pos_tokens.permute(0, 2, 3, 4,
                                    1).reshape(-1, C, T)  # BHW,C,T
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=new_T, mode='linear')
    pos_tokens = pos_tokens.reshape(1, P, P, C,
                                    new_T).permute(0, 4, 1, 2, 3)
    pos_tokens = pos_tokens.flatten(1, 3)
    model.pos_embed = pos_tokens  # update
utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

device = torch.device(args.device)
model.to(device)

input_video = '/home/shk00315/intelligent_robot/VideoMAEv2/dataset/test_dataset_skeleton/bowling_skeleton.mp4'
# 테스트 하고자 하는 데이터셋  추출 
video =  VideoReader(input_video,width=224,height=224)
frame_id_list = range(0,len(video),len(video)//16)
videos = video.get_batch(frame_id_list).asnumpy()
videos= torch.from_numpy(videos.astype(np.float32))
videos = videos.unsqueeze(0)
videos = videos.permute(0,4,1,2,3).to(args.device)
with torch.no_grad() : 
    outputs = model(videos) # outputs값이 softmax넣기 직전의 값이라고 보면됨 

print(outputs)