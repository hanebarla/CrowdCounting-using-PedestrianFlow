import csv
import os

from cv2 import line

myloss = [
    1.0,
    0.1,
    0.01,
    0
]

exp = [
    'activate-relu',
    'no_change',
    'opt-amsgrad'
]

FF_setting = [
    'noFF_result_',
    'DynamicFF_result_',
    'StaticFF_result_',
    'BothFF_result_'
]

scene_set = [
    'A',
    'B',
    'C',
    'D',
    'E'
]

for s in scene_set:
    s_val = []
    s_test = []
    for m in myloss:
        for e in exp:
            val_tmp = []
            test_tmp = []
            for f in FF_setting:
                val_file = os.path.join(
                    '/groups1/gca50095/aca10350zi/habara_exp',
                    'FF_confusion_matrix_{}'.format(s),
                    'CrowdFlow',
                    '{}'.format(m),
                    '{}'.format(e),
                    '{}val.csv'.format(f)
                )
                with open(val_file) as f:
                    reader = csv.reader(f)
                    lines = [row for row in reader]

                val_res = lines[0]
                val_tmp.extend(val_res)

                test_file = os.path.join(
                    '/groups1/gca50095/aca10350zi/habara_exp',
                    'FF_confusion_matrix_{}'.format(s),
                    'CrowdFlow',
                    '{}'.format(m),
                    '{}'.format(e),
                    '{}test.csv'.format(f)
                )
                with open(test_file) as f:
                    reader = csv.reader(f)
                    lines = [row for row in reader]

                test_res = lines[0]
                test_tmp.extend(test_res)

            s_val.append(val_tmp)
            s_test.append(test_tmp)

    with open('./data/{}_val_res.csv'.format(s), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(s_val)

    with open('./data/{}_test_res.csv'.format(s), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(s_test)
