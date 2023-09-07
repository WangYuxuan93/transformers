main=/codes/run_retrieval.py
model=/models/Taiyi-BLIP-full-Chinese
cache=/tmp
train=/data/IC/IC_train.json
valid=/data/IC/IC_val.json
test=/data/IC/IC_test.json
model_type=blip

predict_i2t_result_path=/saves/new_taiyi_blip-itr-base-finetuned/result_i2t_ch_blip.json
predict_t2i_result_path=/saves/new_taiyi_blip-itr-base-finetuned/result_t2i_ch_blip.json

output_dir=/saves/new_taiyi_blip-itr-base-finetuned

python $main \
	--output_dir $output_dir \
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
	--gradient_accumulation_steps="32" \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
	--evaluation_strategy='epoch' \
	--save_strategy='epoch' \
	--logging_strategy='epoch' \
	--learning_rate="1e-5" --warmup_steps="0" --weight_decay 0.05 \
	--lr_scheduler_type cosine \
	--num_train_epochs="15" \
	--push_to_hub &> logs/new_taiyi_blip-itr-base-finetuned.log

