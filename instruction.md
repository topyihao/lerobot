fine-tune: 
CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py   --policy.path=lerobot/smolvla_base   --dataset.repo_id=aloha_put_sponge_into_pot_slow   --dataset.root=/home/aloha/Disk2/lerobot_dataset/aloha_put_sponge_into_pot_slow   --dataset.video_backend=pyav   --policy.device=cuda   --batch_size=64   --steps=50000   --policy.push_to_hub=false   --output_dir=/home/aloha/Disk2/lerobot_checkpoints/smolvla_aloha_sponge_pot_slow --save_freq 1000


