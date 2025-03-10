CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py \
--output_path=checkpoints/ \
--train_csv_path=vietnamese-datasets/metadata_train.csv \
--eval_csv_path=vietnamese-datasets/metadata_eval.csv \
--language="vi" \
--num_epochs=5 \
--batch_size=512 \
--lr=5e-6 \
--wandb_project "xtts-vietnamese-dvae" \
--wandb_run_name "dvae-finetune-run1" \