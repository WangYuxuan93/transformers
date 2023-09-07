main=/codes/run_retrieval.py
model=/saves/new_taiyi_blip-itr-base-finetuned/
cache=/tmp
train=/data/IC/IC_train.json
valid=/data/IC/IC_val.json
test=/data/IC/IC_test.json
model_type=blip

predict_i2t_result_path=/saves/new_taiyi_blip-itr-base-finetuned/result_i2t_ch_blip.json
predict_t2i_result_path=/saves/new_taiyi_blip-itr-base-finetuned/result_t2i_ch_blip.json


python $main \
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
	--do_predict \
	--per_device_eval_batch_size="64" \

