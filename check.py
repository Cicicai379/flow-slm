import torch
from dataset import SpeechDataModule
import yaml
import munch

# load config
with open("conf/1b_extended.yaml") as f:
    conf_dict = yaml.safe_load(f)
conf = munch.munchify(conf_dict)

# simple args placeholder
class Args:
    hf_training_data = True
    training_data = "MLSEn+people"
    train_id_file = None
    data_dir = None

args = Args()
data = SpeechDataModule(args, conf)
data.setup(stage="fit")

# get first batch
batch = next(iter(data.train_dataloader()))
ids, wavs, wav_len = batch[:3]

print("wavs min/max:", wavs.min().item(), wavs.max().item())
print("wavs contains NaN:", torch.isnan(wavs).any().item())
print("wavs contains Inf:", torch.isinf(wavs).any().item())
print("zero-length audio:", (wav_len == 0).any().item())
