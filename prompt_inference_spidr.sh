cfg_scale=0.3
temp=0.8
token_temp=0.8
ode_steps=32
topp=0.95
prompts="prompts/valid_libri_1_sample.csv"
root_dir=/data/cicicai/LibriSpeech
n_quantizers=16
penalize_weight=10
vocoder_type="mimi"
max_len=10
samples_per_prompt=4
batch_size=4
solver="euler"

ckpt_path="/data/cicicai/flow_slm/checkpoints/test_run_spidr/270m_spidr/model-step=0015000.ckpt"

conf_path="conf/270m_spidr.yaml"
output_dir="./test_output_spidr/"

if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

python inference.py --ckpt_path $ckpt_path --conf_path $conf_path --batch_size $batch_size --output_dir $output_dir --temperature $temp --ode_steps $ode_steps --prompt_dir $root_dir --prompt_csv $prompts --solver $solver --samples_per_prompt $samples_per_prompt --save_transcription --save_wav --token_temperature $token_temp --cfg_scale $cfg_scale --num_quantizers $n_quantizers --max_len $max_len --penalize_silence --penalize_weight $penalize_weight
