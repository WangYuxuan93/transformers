
import json
from tqdm import tqdm
import argparse

def compute_VG(ref_list,pre_list):

    assert len(ref_list) == len(pre_list)

    scores = []
    correct = 0
    total = len(ref_list)
    n = 0

    for line in pre_list:
        for new_line in ref_list:
            if line['image'] == new_line['image'] and line['ref_id'] == new_line['ref_id'] and line['text'] == new_line['text']:

                n += 1

                gold_bbox = new_line['bbox']
                pred_bbox = line['bbox']

                x1 = max(gold_bbox[0], pred_bbox[0])
                y1 = min(gold_bbox[1], pred_bbox[1])
                x2 = min(gold_bbox[0] + gold_bbox[2] , pred_bbox[0] + pred_bbox[2] )
                y2 = max(gold_bbox[1] - gold_bbox[3] , pred_bbox[1] - pred_bbox[3] )

                if x1 < x2 and y1 > y2:
                    inter = (x2 - x1) * (y1 - y2 )
                else:
                    inter = 0

                union = gold_bbox[2] * gold_bbox[3] + pred_bbox[2] * pred_bbox[3] - inter

                score = float(inter) / union

                if score > 0.5:
                    correct += 1

                scores.append(score)

    return scores,float(correct) / float(n)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--ref_path',type=str, help='The path of reference file.')
    parser.add_argument('--pre_path',type=str, help='The path of prediction file.')
    args = parser.parse_args()

    ref_path = args.ref_path
    pre_path = args.pre_path
    ref = []
    pre = []

    with open(ref_path, 'r', encoding='utf-8') as f:
        string = f.read()
        raw_data = json.loads(string)
        for line in tqdm(raw_data):
            ref.append(line)

    with open(pre_path, 'r', encoding='utf-8') as f:
        string = f.read()
        raw_data = json.loads(string)
        for line in tqdm(raw_data):
            pre.append(line)

    IOU,result = compute_VG(ref,pre)
    print('IOU:',sum(IOU) / len(IOU))
    print('VG_score:',result)