
import json
from tqdm import tqdm
import argparse

def R_n(pres,refs,n):

    assert len(pres) == len(refs)

    total = len(pres)
    correct = 0
    m = 0
    for pre_lines in pres:

        for ref_lines in refs:

            if pre_lines['image'] == ref_lines['image']:
                m += 1
                gold_caps = ref_lines['caption']
                pre_caps = pre_lines['caption']

                pre_index = 20
                for cap in pre_caps:
                    if cap in gold_caps:
                        new_pre_index = pre_caps.index(cap)
                        if new_pre_index < pre_index:
                            pre_index = new_pre_index

                if pre_index < n:
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

    R_1 = R_n(pre,ref,1)
    R_2 = R_n(pre,ref,2)
    R_5 = R_n(pre,ref,5)
    R_10 = R_n(pre, ref, 10)

    print('R_1:',R_1,'  R_2:', R_2, '   R_5:',R_5,  'R_10:',R_10)