import os
import re
import pickle
import sys

# def clean_str(string):
# 	string = re.sub(r"\s{2,}", " ", string)
# 	return string.strip().lower()

def rewrite(src_file, tgt_list, label):
	with open(src_file, 'r') as f:
		for line in f:
			# clean_line = clean_str(line)
			# tgt_list.append([clean_line, label])
			tgt_list.append([line.strip(), label])


def main(dataset):
	data_dir = os.path.join('../sata/datasets', dataset)
	src_dir = os.path.join(data_dir, 'corpus')
	tgt_dir = os.path.join(data_dir, 'processed')
	if not os.path.exists(tgt_dir):
		os.makedirs(tgt_dir)
	label_dict = ['<NEG>', '<POS>'] if dataset == 'yelp' or dataset == 'amazon' else ['<FORMAL>', '<INFORMAL>']
	result = dict()

	for split in ('train', 'dev', 'test'):
		split_list = []
		for i in range(2):
			split_file = os.path.join(src_dir, '{}.{}'.format(split, i))
			rewrite(split_file, split_list, label_dict[i])
		result[split] = split_list
		print('{} size: {}'.format(split, len(split_list)))

	tgt_file = os.path.join(tgt_dir, '{}.pkl'.format(dataset))
	with open(tgt_file, 'wb') as f:
		pickle.dump(result, f, -1)

if __name__ == '__main__':
	main(sys.argv[1])






