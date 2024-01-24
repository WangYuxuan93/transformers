model="/home/mrch/transformers/examples/pytorch/image-pretraining/outputs_mae/checkpoint-201192"
data="/home/mrch/data/CelebAMask-HQ/celeba_mask"
CUDA_VISIBLE_DEVICES=7 CUDA_LAUNCH_BLOCKING=1 python run_semantic_segmentation_for_vit.py \
	--model_name_or_path $model \
	--model_type vitmae \
	--dataset_name $data \
	--output_dir ./vitmae_outputs/ \
	--remove_unused_columns False \
	--do_train \
	--do_eval \
	--evaluation_strategy steps \
	--max_steps 10000 \
	--learning_rate 0.00006 \
	--lr_scheduler_type polynomial \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 64 \
	--logging_strategy steps \
	--logging_steps 100 \
	--eval_steps 100 \
	--evaluation_strategy steps \
	--save_strategy epoch \
	--seed 1337
