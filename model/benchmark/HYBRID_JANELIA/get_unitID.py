import numpy as np
import os

class UnitIDManager:
    def __init__(self):
        # This is a dictionary that stores the unit IDs for each recording in each study
        # only include the units that SNR > 3
        self.study_data = {
            'hybrid_static_tetrode': {
                'rec_4c_600s_11': [1, 7, 13, 20, 21, 25, 52, 56, 62, 63, 68],
                'rec_4c_600s_12': [1, 7, 13, 20, 21, 25, 52, 56, 62, 63, 68],
                'rec_4c_600s_21': [12, 14, 19, 30, 32, 42, 57, 60, 65, 66, 69, 70, 71, 72, 73],
                'rec_4c_600s_22': [12, 14, 19, 30, 32, 42, 54, 57, 60, 65, 66, 69, 70, 73],
                'rec_4c_600s_31': [18, 23, 26, 27, 29, 31, 34, 35, 36, 37, 49, 55, 61, 64],
                'rec_4c_600s_32': [18, 23, 26, 27, 29, 31, 34, 35, 36, 37, 49, 55, 61, 64],
                'rec_4c_1200s_11': [1, 7, 13, 20, 21, 25, 52, 56, 62, 63, 68],
                'rec_4c_1200s_21': [19, 30, 32, 42, 57, 60, 65, 66, 69, 70, 72, 73],
                'rec_4c_1200s_31': [18, 23, 26, 27, 28, 31, 35, 36, 37, 49, 55, 61, 64],

            },
            'hybrid_drift_tetrode': {
                'rec_4c_600s_11': [1, 7, 13, 20, 21, 25, 52, 56, 62, 63, 68],
                'rec_4c_600s_12': [1, 7, 13, 20, 21, 25, 52, 56, 62, 63, 68],
                'rec_4c_600s_21': [12, 14, 19, 30, 32, 42, 54, 57, 60, 65, 66, 69, 70, 73],
                'rec_4c_600s_22': [12, 14, 19, 30, 32, 42, 54, 57, 60, 65, 66, 69, 70, 73],
                'rec_4c_600s_31': [18, 23, 26, 27, 29, 31, 34, 35, 36, 37, 49, 55, 61, 64],
                'rec_4c_600s_32': [18, 23, 26, 27, 29, 31, 34, 35, 36, 37, 49, 55, 61, 64],
                'rec_4c_1200s_11': [1, 7, 13, 20, 21, 25, 52, 56, 62, 63, 68],
                'rec_4c_1200s_21': [19, 30, 32, 42, 57, 60, 65, 66, 69, 70, 72, 73],
                'rec_4c_1200s_31': [18, 23, 26, 27, 28, 31, 35, 36, 37, 49, 55, 61, 64],

            },

            # 'another_study_name': {
            #     'rec_abc': [...],
            #     'rec_xyz': [...]
            # }
        }

    def __call__(self, study_name, record_name):

        study = self.study_data.get(study_name, None)
        if study:
            return study.get(record_name, None)
        return None

if __name__ == '__main__':

    IDs = UnitIDManager()
    unit_ids_600s_21 = IDs('hybrid_static_tetrode', 'rec_4c_600s_21')

    print(f'unit_ids for rec_4c_600s_21: {unit_ids_600s_21}')


