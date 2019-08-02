# Audio:
num_mels = 80
num_freq = 1025
sample_rate = 22050

frame_length_ms = 50
frame_shift_ms = 12.5

hop_length = 256
win_length = 1024

preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5
signal_normalization = True
use_lws = False

# num_mels = 80
outputs_per_step = 1
hidden_size = 256
embedding_size = 512
epochs = 10000
lr = 0.001
save_step = 2000
batch_size = 16
cleaners = ['english_cleaners']
data_path = './dataset'
checkpoint_path = './model_new'
logger_path = "./logger"
log_step = 10
clear_Time = 20
