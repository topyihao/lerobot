data converter:
python src/lerobot/scripts/aloha_dataset_converter.py

fine-tune: 
CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py   --policy.path=lerobot/smolvla_base   --dataset.repo_id=aloha_put_sponge_into_pot_slow   --dataset.root=/home/aloha/Disk2/lerobot_dataset/aloha_put_sponge_into_pot_slow   --dataset.video_backend=pyav   --policy.device=cuda   --batch_size=64   --steps=50000   --policy.push_to_hub=false   --output_dir=/home/aloha/Disk2/lerobot_checkpoints/smolvla_aloha_sponge_pot_slow --save_freq 10000


python -m lerobot.scripts.train --policy.path=lerobot/pi0fast_base --dataset.repo_id=aloha_clean_dish  --dataset.root=/home/allied/Disk2/Yihao/lerobot_dataset/aloha_clean_dish   --dataset.video_backend=pyav   --policy.device=cuda --policy.push_to_hub=false --output_dir=/home/allied/Disk2/Yihao/checkpoints/pi_0_fast_clean_dish/ --batch_size=8 --steps=100000 --save_freq 10000 --wandb.project=lerobot --wandb.entity=topyihaozhang --wandb.notes="pi0fast clean_dish"