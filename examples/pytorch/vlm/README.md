# 基于多个模型实现中文图文多模态任务的示例

以下的示例演示如何使用预训练模型，完成4个中文图文多模态任务。具体来说，预训练模型有 clip 、blip 、chinese_clip ，4个下游任务分别是图文检索、图片对话、图片对齐和图片问答。

# 预训练模型选择

对于图文检索任务，使用的预训练模型如下：

| 模型 | 基座模型 |
| :------: | :------: |
| blip | Taiyi-BLIP-full-Chinese |
| clip | taiyi-clip-roberta-new |
| chinese_clip | chinese-clip-vit-base-patch16 |

对于其他三个任务，使用的预训练模型如下：

<table>
    <tr >
	<td align="center"> <b>模型</b></td>
	<td align="center"><b>text-encoder</b></td>
	<td align="center"><b>vision-encoder</b></td> 
    <td align="center"><b>text-decoder</b></td>
	</tr>
    <tr>
    <td align="center">blip</td>
    <td colspan="3" align="center"> Taiyi-BLIP-full-Chinese </td>
    </tr>
    <tr>
    <td align="center">clip</td>
    <td colspan="2" align="center">taiyi-clip-roberta-new-for-vqa</td>
    <td align="center">taiyi-clip-roberta-new-for-vqa</td>
    </tr>
    <tr>
    <td align="center">chinese_clip</td>
    <td colspan="2" align="center">chinese-clip-vit-base-patch16-for-vqa</td>
    <td align="center">Taiyi-CLIP-Roberta-102M-Chinese </td>
    </tr>
</table> 

在代码中，也会根据设定具体参数```--model_type```和下游任务加载特定的预训练模型参数，下面用图片问答任务举例。

```python
image_processor = AutoImageProcessor.from_pretrained(
    model_args.image_processor_name or model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

if model_args.model_type == 'clip':
    model = CLIPForVQA.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        decoder_model_path= model_args.decoder_model_path,
        use_auth_token=True if model_args.use_auth_token else None,
    )

elif model_args.model_type == 'blip':
    model = BlipForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

elif model_args.model_type == 'chinese':
    model = ChineseCLIPForVQA.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        decoder_model_path= model_args.decoder_model_path,
        use_auth_token=True if model_args.use_auth_token else None,
    )

config = model.config
```

## 预处理

按比例划分训练集、验证集和测试集（6:1:3）。然后，对数据进行读取，依据不同特征进行存储。之后，再按照常规方法对所需的图片和文本特征进行处理后，传入模型。

不同任务，传入模型的变量也有所不同。接下来，针对不同任务，给出详细的预处理过程。

### 图文检索

图文检索的输入比较简单，只需要图片和描述的基本信息。只利用最基本的操作即可完成，无需更多的处理，这里也不展开叙述。

### 图片对话

由于在对话任务中，我们需要对对话过程的历史信息进行存储。这里，我们选择将前一轮对话的历史信息加上前一轮对话的内容，作为当前对话的历史信息。而对于第一轮对话，则选取图片的描述作为第一轮对话的历史信息。这样，通过迭代的方式构造每一轮对话的历史信息。具体实现如下（在实际训练时，可根据需要进行修改）：

```python
if dialog_id == 0:
    his = captions[i] + ' ' + sep + ' ' + questions[i]
    last_his =  his

    his_inputs = tokenizer(his, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
    text_inputs_ids.append(his_inputs.input_ids)
    text_attention_mask.append(his_inputs.attention_mask)
else:
    his = last_his + ' ' + sep + ' ' + answers[i-1] + ' ' + sep + ' ' + questions[i]
    if len(his) > data_args.max_seq_length - 2:
        his = his[-(data_args.max_seq_length - 2):]
    last_his =  his

    his_inputs = tokenizer(his, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
    text_inputs_ids.append(his_inputs.input_ids)
    text_attention_mask.append(his_inputs.attention_mask)
```

### 图片对齐

在图片对齐任务中，由于图片的规格并不一致，需要先将图片的长宽按照设定的图片尺寸进行缩放，然后，将坐标转化成比例的形式。此外，由于初始给定的是左下角坐标，不便于模型的学习。这里，还需要将其转换为中心坐标的形式。具体实现如下：

```python
def trans_bbox(bbox, width, height):
    x , y , w , h = bbox

    new_x = x / width * new_image_size
    new_y = y / height * new_image_size
    new_w = w / width * new_image_size
    new_h = h / height * new_image_size

    center_x = new_x + new_w / 2
    center_y = new_y - new_h / 2

    return torch.tensor([center_x / new_image_size, center_y / new_image_size, new_w / new_image_size, new_h / new_image_size], dtype=torch.float)
```

### 图片问答

图片问答的预处理内容与图文检索十分相似，只不过图片问答任务中，文本信息不是描述，而是问题、答案及答案 列表，其余部分和图文检索一致。

## 精调

接下来对预训练模型，针对具体的下游任务进行精调。在本项目中，精调部分的代码在 ```run_×××.py``` 中，通过调用相关任务的具体模型的shell脚本，即可完成相关参数的设定与程序的运行。

下面也同样用图片问答任务进行举例，具体选用的是 clip 模型，通过调用 ```run_vqa_clip.sh``` 完成预测。

    python /codes/run_vqa.py \
        --output_dir /saves/taiyi_clip-vqa-base-finetuned \
        --model_name_or_path /models/taiyi-clip-roberta-new-for-vqa \
        --decoder_model_path /models/Taiyi-CLIP-Roberta-102M-Chinese \
        --model_type clip \
        --cache_dir $cache \
        --train_file $train \
        --validation_file $valid \
        --answer_list $answer_list \
        --remove_unused_columns=False \
        --label_names answer_input_ids \
        --do_train  \
        --do_eval \
        --evaluation_strategy='epoch' \
        --save_strategy='epoch' \
        --logging_strategy='epoch' \
        --gradient_accumulation_steps="8" \
        --per_device_train_batch_size="32" \
        --per_device_eval_batch_size="32" \
        --learning_rate="2e-5" --warmup_steps="0" --weight_decay 0.05 \
        --lr_scheduler_type cosine \
        --num_train_epochs="10" \
        --push_to_hub &> logs/taiyi_clip-vqa-base-finetuned.log

## 预测

训练得到的模型保存在文件夹 ```/saves``` 下，选取在开发集上效果最好的一组参数，作为预测时使用的模型参数。

在预测时，预测部分的代码也同样在 ```run_×××.py``` 中。通过调用相关任务的具体模型预测的shell脚本，即可完成相关参数的设定与程序的运行。

下面也同样用图片问答任务进行举例，具体选用的是 clip 模型，通过调用 ```pred_vqa_clip.sh``` 完成预测。

    python /codes/run_vqa.py \
        --model_name_or_path $model \
        --decoder_model_path /models/Taiyi-CLIP-Roberta-102M-Chinese \
        --model_type clip \
        --test_file $test \
        --answer_list $answer_list \
        --predict_result_path $predict_result_path \
        --remove_unused_columns=False \
        --label_names answer_input_ids \
        --do_predict \
        --per_device_eval_batch_size="256" \

## 指标评测

前面完成了预训练模型的精调和预测，下面，可以对模型的效果进行一些特定指标的评测。接下来，对不同任务分别进行阐述。

### 图文检索

针对图文检索任务，评测指标选用召回 r@k（k取1，5，10）。在本项目中，直接调用 IR.py 与 TR.py 分别完成 image-to-text 和 text-to-image 相关指标的计算。

### 图片对话

针对图片对话任务，评测指标选用对话中标准回答在预测排名中的平均排名、召回 r@k （k取1，5，10）和平均倒数排名（MRR）作为评价指标。在本项目中，直接调用 VD.py 完成相关指标的计算。

### 图片对齐

针对图片对齐任务，评测指标选用图片对齐的准确率（框选范围的 IoU 值大于0.5视为正确）和 IoU 的均值。在本项目中，直接调用 VG.py 完成相关指标的计算。

### 图片问答

针对图片问答任务，评测指标比较简单，即回答问题的准确率。在本项目中，直接调用 VQA.py 完成相关指标的计算。

### 运行示例

四个任务的评测方法完全相同，评测代码文件也在上面给出，为了方便运行，

## 文件说明

在 ```vlm``` 目录下，有5个文件夹： ```codes``` 存储各任务实现的代码，其子目录下保存着各模型运行的脚本； ```data``` 存储各任务的数据； ```models``` 存储着预训练模型的相关参数； ```saves``` 用来保存精调后的模型参数和预测结果； ```tmp``` 是一个保存临时文件的文件夹。