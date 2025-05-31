WANDB_MODE=online WANDB_RUN_ID=hres-med python main.py --project hres --mode train \
--cuda_opt \
--config configs/ve/single_matryoshka_config.py --workdir remote_workdir/hres/frozen-med \
--config.data.dir_path /DATA/Users/amahmood/braintyp/processed_v2/ \
--config.data.splits_dir /codespace/sade/sade/datasets/brains/ \
--config.training.batch_size=2 \
--config.eval.batch_size=4 --config.eval.sample_size=2 \
--config.training.log_freq=50 --config.training.eval_freq=100 \
--config.optim.warmup=10_000 --config.optim.lr=1e-5 \
--config.model.fourier_scale=4 --config.model.time_embedding_sz=128 \
--config.model.trainable_inner_model=False \
--config.training.sampling_freq=50_000 \
--config.data.cache_rate=0

WANDB_MODE=online WANDB_RUN_ID="warmed-v0" CUDA_VISIBLE_DEVICES=1 python main.py \
--project flows --mode flow-train --config configs/flows/matryoshka_gmm_flow_config.py --workdir remote_workdir/hres/frozen-med \
--config.model.fourier_scale=4 --config.model.time_embedding_sz=128 \
--config.model.trainable_inner_model=False \
--config.training.pretrained_checkpoint=/ASD/ahsan_projects/braintypicality/workdir/hres/frozen-med/checkpoints/checkpoint_60.pth \
--config.flow.patches_per_train_step=16384 --config.flow.patch_batch_size=16384 \
--config.flow.training_kimg=200 \
--config.flow.training_fast_mode=True 
# --config.data.dir_path=/BEE/Connectome/ABCD/Users/amahmood/braintyp/processed_v2/

