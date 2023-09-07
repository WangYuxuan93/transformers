main=/codes/run_retrieval.py
model=/models/taiyi-clip-roberta
output_dir=/saves/taiyi_clip-itr-base-finetuned
cache=/tmp
test=/data/IC/IC_test.json
model_type=clip

predict_i2t_result_path=/saves/taiyi_clip-itr-base-finetuned/result_i2t_ch_clip.json
predict_t2i_result_path=/saves/taiyi_clip-itr-base-finetuned/result_t2i_ch_clip.json

python $main \
	--output_dir $output_dir \
	--model_name_or_path $model \
	--model_type $model_type \
    --cache_dir $cache \
	--test_file $test \
	--predict_i2t_result_path $predict_i2t_result_path \
	--predict_t2i_result_path $predict_t2i_result_path \
    --image_column image \
    --caption_column caption \
	--remove_unused_columns=False \
	--do_predict \
	--per_device_eval_batch_size="256" \

