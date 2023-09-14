
import json
from tqdm import tqdm
import argparse

def compute_VQA(pres,refs):

    assert len(pres) == len(refs)

    correct = 0
    total = len(pres)

    for pre_line in pres:
        for ref_line in refs:
            if pre_line['question_id'] == ref_line['question_id']:

                if pre_line['answer'] == ref_line['answer']:
                    correct += 1

    return float(correct) / float(total)

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

    result = compute_VQA(pre,ref)

    print('VQA_score:',result)