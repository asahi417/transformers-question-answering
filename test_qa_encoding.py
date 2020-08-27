""" Test qa encoding """

import os

from get_dataset import get_dataset_qa
from transformers_qa import Transforms

CACHE_DIR = os.getenv("CACHE_DIR", './cache')
TRANSFORM = Transforms('xlm-roberta-base')


def test(name, v2_with_negative):
    print(name)
    _data = get_dataset_qa(name, CACHE_DIR, v2_with_negative)
    print("### encode test ### ")
    fail_list = []
    for _type in ['train', 'valid']:
        print(_type)
        negatives = 0
        for i in range(len(_data[_type]['context'])):
            answer = _data[_type]['answer'][i]
            context = _data[_type]['context'][i]
            question = _data[_type]['question'][i]
            question_id = _data[_type]['question_id'][i]
            _encode = TRANSFORM.encode_plus(
                context=context,
                answer=answer,
                question=question,
                question_id=question_id,
                with_negative=v2_with_negative,
                return_meta_data=True)
            # print('\n- encoded data')
            # if is_impossible:
            #     print(_encode)
            if len(_encode) == 0:
                print('\n- original data')
                print(context)
                print(answer)
                print(question)
                input('unknown error happens>>>')
            else:
                for e in _encode:
                    if e['is_impossible']:
                        negatives += 1
                    else:
                        start, end = answer
                        org_answer = context[start:end]
                        _start = e['start_positions']
                        _end = e['end_positions']
                        decode_answer = TRANSFORM.tokenizer.decode(e['input_ids'][_start:_end])
                        if org_answer != decode_answer:
                            fail_list.append([_type, org_answer, decode_answer])
                            print('WARNING: {0} || {1}'.format(org_answer, decode_answer))
        print('{} negatives'.format(negatives))
    print('{} entries failed'.format(len(fail_list)))
    print(TRANSFORM.encode_tensor_list)


if __name__ == '__main__':
    test('squad-v2', False)
    # test('squad-v2', True)
    # test('squad-v2')
