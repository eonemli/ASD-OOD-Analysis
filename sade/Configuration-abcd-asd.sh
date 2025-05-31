



CUDA_VISIBLE_DEVICES=2 WANDB_MODE=online WANDB_RUN_ID=hres-med \
python main.py --project hres --mode train \
--config configs/ve/single_matryoshka_config.py \
--workdir remote_workdir/hres/frozen-med \
--cuda_opt \
--config.data.dir_path /DATA/processed/ \
--config.data.splits_dir /path/tp/splits/dir/ \
--config.training.batch_size=2 \
--config.eval.batch_size=4 --config.eval.sample_size=2 \
--config.training.log_freq=50 --config.training.eval_freq=100 \
--config.optim.warmup=10_000 --config.optim.lr=1e-5 \
--config.training.sampling_freq=50_000 \
--config.model.trainable_inner_model=False \
--config.data.cache_rate=0s





CUDA_VISIBLE_DEVICES=3 \
python main.py --mode inference \
--config configs/flows/gmm_flow_config.py \
--workdir /workdir/cuda_opt/learnable/ \
--config.data.dir_path /DATA/Users/emre/braintyp/processed_v3/ \
--config.data.splits_dir /braintypicality-scripts/split-keys-abcd-asd \
--config.eval.checkpoint_num=150 \
--config.eval.experiment.flow_checkpoint_path=/workdir/cuda_opt/learnable/flow/psz3-globalpsz17-nb20-lr0.0003-bs32-np1024-kimg300_smin1e-2_smax0.8 \
--config.msma.min_timestep=0.01 \
--config.msma.max_timestep=0.8 \
--config.eval.experiment.train=abcd-train  \
--config.eval.experiment.inlier=abcd-val \
--config.eval.experiment.ood=abcd-asd \
—-config.eval.experiment.id=abcd-asd-161-v1





```
No, I forgot to run the inference for abcd-train. I am pasting the command line you can use with sade. The inference file will look for train, inlier and ood dataset names under config.eval.experiment. You can just cancel the script once it finishes with abcd-train as the others are already computed. Also, make sure to pull in updates from github if you haven’t.


python main.py --mode inference \
--config configs/flows/gmm_flow_config.py \
--workdir remote_workdir/cuda_opt/learnable/ \
--config.eval.checkpoint_num=150 \
--config.eval.experiment.flow_checkpoint_path=remote_workdir/cuda_opt/learnable/flow/psz3-globalpsz17-nb20-lr0.0003-bs32-np1024-kimg300_smin1e-2_smax0.8 \
--config.msma.min_timestep=0.01 \
--config.msma.max_timestep=0.8 \
--config.eval.experiment.train=abcd-train  \
--config.eval.experiment.inlier=abcd-val \
--config.eval.experiment.ood=ibis-asd \
—config.eval.experiment.id=whatever-you-want


Re inlier concatenation:

Currently you have two copies of abcd_val (you already grabbed one copy from abcd-train_abcd-val_lesion_load_20-enhanced_results.npz). But this should not be a problem now when you load samples from the folders directly.

Also, note that you have abcd-test and ibis-inliers as well that you can use for training the SOMs.
```