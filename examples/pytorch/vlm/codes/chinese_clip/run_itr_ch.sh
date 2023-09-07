main=/codes/run_retrieval.py
model=/models/chinese-clip-vit-base-patch16 
cache=/tmp 
train=/data/IC/IC_train.json
valid=/data/IC/IC_val.json
test=/data/IC/IC_test.json
model_type=chinese

predict_i2t_result_path=/saves/chinese_clip-itr-base-finetuned/result_i2t_ch_chinese.json
predict_t2i_result_path=/saves/chinese_clip-itr-base-finetuned/result_t2i_ch_chinese.json

output_dir=/saves/chinese_clip-itr-base-finetuned

python $main 
	--output_dir  $output_dir \
	--model_name_or_path $model \
	--model_type $model_type \
    --cache_dir $cache \
	--train_file $train \
	--validation_file $valid \
	--test_file $test \
	--predict_i2t_result_path $predict_i2t_result_path \
	--predict_t2i_result_path $predict_t2i_result_path \
    --image_column image \
    --caption_column caption \
	--remove_unused_columns=False \
	--do_train  \
	--do_eval \
	--do_predict \
	--gradient_accumulation_steps="4" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="64" \
	--evaluation_strategy='epoch' \
	--save_strategy='epoch' \
	--logging_strategy='epoch' \
	--learning_rate="1e-5" --warmup_steps="0" --weight_decay 0.05 \
	--lr_scheduler_type cosine \
	--overwrite_output_dir \
	--num_train_epochs="6" \
	--push_to_hub &> logs/chinese_clip-itr-base-finetuned.log

