import sys
import pandas as pd
import simpledorff
import warnings
warnings.filterwarnings('ignore')

filename = sys.argv[1]

ann = pd.read_csv(filename)
ann_1 = ann[['tweet_ID', 'annotator_1']]
ann_2 = ann[['tweet_ID', 'annotator_2']]
ann_3 = ann[['tweet_ID', 'annotator_3']]
ann_1.rename(columns={'annotator_1': 'annotation'}, inplace = True)
ann_2.rename(columns={'annotator_2': 'annotation'}, inplace = True)
ann_3.rename(columns={'annotator_3': 'annotation'}, inplace = True)
ann_1['annotator_id'] = 'Rajiv'
ann_2['annotator_id'] = 'Ishansh'
ann_3['annotator_id'] = 'Shreya'
final_arr = [ann_1, ann_2, ann_3]
final_df = ann_1.append(ann_2, ignore_index=True)
final_df = final_df.append(ann_3, ignore_index=True)
final_df.rename(columns={'tweet_ID': 'document_id'}, inplace = True)

alpha_value = simpledorff.calculate_krippendorffs_alpha_for_df(final_df,experiment_col='document_id',
                                                 annotator_col='annotator_id',
                                                 class_col='annotation')

ann.replace(to_replace = 'X', value = 2, inplace = True)
ann[['annotator_1', 'annotator_2', 'annotator_3']] = ann[['annotator_1', 'annotator_2', 'annotator_3']].apply(pd.to_numeric)
ann['Difference'] = (ann['annotator_1'].fillna(0) - ann['annotator_2'].fillna(0) - ann['annotator_3'].fillna(0))
percent_agreement = (ann['Difference'].value_counts()[0.0]/len(ann['Difference']))*100
print(alpha_value, '\t', percent_agreement)