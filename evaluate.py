import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import time
import matplotlib as mpl
import copy
import os
# mpl.use('Agg')
# import matplotlib.pyplot as plt
import os
# Number of Epochs
epochs = 100
# Batch Size
batch_size = 128
# Time Steps of Input, f = 6 skeleton frames
time_steps = 6
# Length of Series, J = 20 body joints in a sequence
series_length = 20
# Learning Rate
learning_rate = 0.0005
dataset = False
attention = False
gpu = False
manner = False


tf.app.flags.DEFINE_string('attention', 'LA', "(LA) Locality-aware Attention or BA (Basic Attention)")
tf.app.flags.DEFINE_string('manner', 'ap', "sequence-level concatenation (sc) or average prediction (ap)")
tf.app.flags.DEFINE_string('dataset', 'BIWI', "Dataset: BIWI or IAS or KGBD")
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
FLAGS = tf.app.flags.FLAGS

def main(_):
	global attention, dataset, series_length, epochs, time_steps, gpu, manner
	attention, dataset, gpu, manner = FLAGS.attention, FLAGS.dataset, FLAGS.gpu, FLAGS.manner
	if attention not in ['BA', 'LA']:
		raise Exception('Attention must be BA or LA')
	if manner not in ['sc', 'ap']:
		raise Exception('Training manner must be sc or ap')
	if dataset not in ['BIWI', 'IAS', 'KGBD']:
		raise Exception('Dataset must be BIWI, IAS or KGBD')
	if not gpu.isdigit() or int(gpu) < 0:
		raise Exception('GPU number must be a positive integer')
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	folder_name = dataset + '_' + attention
	series_length=20
	time_steps = 6
	# epochs = 400
	# print('Print the Validation Loss and Rank-1 Accuracy for each testing bacth: ')
	evaluate_reid('./Models/AGEs_RN_models/' + dataset + '_' + attention + '_RN_' + manner)

def get_new_train_batches(targets, sources, batch_size):
	if len(targets) < batch_size:
		yield targets, sources
	else:
		for batch_i in range(0, len(sources) // batch_size):
			start_i = batch_i * batch_size
			sources_batch = sources[start_i:start_i + batch_size]
			targets_batch = targets[start_i:start_i + batch_size]
			yield targets_batch, sources_batch

def evaluate_reid(model_dir):
	global batch_size, dataset, manner
	X = np.load(model_dir + '/val_X.npy')
	y = np.load(model_dir + '/val_y.npy')
	if dataset == 'IAS':
		X_2 = np.load(model_dir + '/val_2_X.npy')
		y_2 = np.load(model_dir + '/val_2_y.npy')
	if dataset == 'BIWI':
		classes = [i for i in range(28)]
	elif dataset == 'KGBD':
		classes = [i for i in range(164)]
	else:
		classes = [i for i in range(11)]
	checkpoint = model_dir + "/trained_model.ckpt"
	loaded_graph = tf.get_default_graph()
	from sklearn.preprocessing import label_binarize
	from sklearn.metrics import roc_curve, auc
	def cal_AUC(score_y, pred_y, ps, draw_pic=False):
		score_y = np.array(score_y)
		pred_y = label_binarize(np.array(pred_y), classes=classes)
		# Compute micro-average ROC curve and ROC area
		fpr, tpr, thresholds = roc_curve(pred_y.ravel(), score_y.ravel())
		roc_auc = auc(fpr, tpr)
		print(ps + ': ' + str(roc_auc))
		if draw_pic:
			fig = plt.figure()
			lw = 2
			plt.plot(fpr, tpr, color='darkorange',
			         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Receiver operating characteristic: ' + ps)
			plt.legend(loc="lower right")
			fig.savefig('30 epoch ROC')
			plt.close()
	with tf.Session(graph=loaded_graph) as sess:
		loader = tf.train.import_meta_graph(checkpoint + '.meta')
		loader.restore(sess, checkpoint)
		X_input = loaded_graph.get_tensor_by_name('X_input:0')
		y_input = loaded_graph.get_tensor_by_name('y_input:0')
		lr = loaded_graph.get_tensor_by_name('learning_rate:0')
		pred = loaded_graph.get_tensor_by_name('add_1:0')
		cost = loaded_graph.get_tensor_by_name('new_train/Mean:0')
		accuracy = loaded_graph.get_tensor_by_name('new_train/Mean_1:0')
		correct_num = 0
		total_num = 0
		rank_acc = {}
		ys = []
		preds = []
		accs = []
		cnt = 0
		if dataset == 'IAS':
			print('Validation Results on IAS-A: ')
		if manner == 'sc':
			for batch_i, (y_batch, X_batch) in enumerate(
				get_new_train_batches(y, X, batch_size)):
				loss, acc, pre = sess.run([cost, accuracy, pred],
				                   {X_input: X_batch,
				                    y_input: y_batch,
				                    lr: learning_rate})
				ys.extend(y_batch.tolist())
				preds.extend(pre.tolist())
				accs.append(acc)
				cnt += 1
				for i in range(y_batch.shape[0]):
					for K in range(1, len(classes) + 1):
						if K not in rank_acc.keys():
							rank_acc[K] = 0
						t = np.argpartition(pre[i], -K)[-K:]
						if np.argmax(y_batch[i]) in t:
							rank_acc[K] += 1
				correct_num += acc * batch_size
				total_num += batch_size
				print(
					'Testing Bacth: {:>3} - Validation Loss: {:>6.3f} - Validation Rank-1 Accuracy {:>6.3f}'
						.format(cnt,
					            loss,
					            acc,
					            ))
			total_acc = correct_num / total_num
			print('Rank-1 Accuracy: %f' % total_acc)
			cal_AUC(score_y=preds,pred_y=ys, ps='nAUC')
		else:
			all_frame_preds = []
			for batch_i, (y_batch, X_batch) in enumerate(
				get_new_train_batches(y, X, batch_size)):
				loss, acc, pre = sess.run([cost, accuracy, pred],
				                   {X_input: X_batch,
				                    y_input: y_batch,
				                    lr: learning_rate})
				ys.extend(y_batch.tolist())
				preds.extend(pre.tolist())
				all_frame_preds.extend(pre)
				accs.append(acc)
				cnt += 1
				# for i in range(y_batch.shape[0]):
				# 	for K in range(1, len(classes) + 1):
				# 		if K not in rank_acc.keys():
				# 			rank_acc[K] = 0
				# 		t = np.argpartition(pre[i], -K)[-K:]
				# 		if np.argmax(y_batch[i]) in t:
				# 			rank_acc[K] += 1
				# correct_num += acc * batch_size
				# total_num += batch_size
				# print(
				# 	'Testing Bacth: {:>3} - Validation Loss: {:>6.3f} - Validation Rank-1 Accuracy {:>6.3f}'
				# 		.format(cnt,
				# 	            loss,
				# 	            acc,
				# 	            ))
			sequence_pred_correct = 0
			sequence_num = 0
			sequence_preds = []
			sequence_ys = []
			for k in range(len(all_frame_preds) // time_steps):
				sequence_labels = np.argmax(y[k * time_steps: (k + 1) * time_steps], axis=1)
				# print(sequence_labels)
				if (sequence_labels == np.tile(sequence_labels[0], [sequence_labels.shape[0]])).all():
					frame_predictions = np.array(all_frame_preds[k * time_steps: (k + 1) * time_steps])
					sequence_pred = np.argmax(np.average(frame_predictions, axis=0))
					if sequence_pred == sequence_labels[0]:
						sequence_pred_correct += 1
					sequence_num += 1
					sequence_ys.append(sequence_labels[0])
					aver = np.average(frame_predictions, axis=0)
					sequence_preds.append(aver)
			seq_acc_t = sequence_pred_correct / sequence_num
			# total_acc = correct_num / total_num
			# print('(Frame) Rank-1 Accuracy: %f' % total_acc)
			print('Rank-1 Accuracy: %f' % seq_acc_t)
			sequence_ys = label_binarize(sequence_ys, classes=classes)
			# cal_AUC(score_y=preds,pred_y=ys, ps='nAUC')
			cal_AUC(score_y=sequence_preds, pred_y=sequence_ys, ps='nAUC')
		if dataset == 'IAS':
			print('Validation Results on IAS-B: ')
			# IAS-B
			if manner == 'sc':
				correct_num = 0
				total_num = 0
				rank_acc = {}
				ys = []
				preds = []
				accs = []
				cnt = 0
				for batch_i, (y_batch, X_batch) in enumerate(
						get_new_train_batches(y_2, X_2, batch_size)):
					loss, acc, pre = sess.run([cost, accuracy, pred],
					                          {X_input: X_batch,
					                           y_input: y_batch,
					                           lr: learning_rate})
					ys.extend(y_batch.tolist())
					preds.extend(pre.tolist())
					accs.append(acc)
					cnt += 1
					# for i in range(y_batch.shape[0]):
					# 	for K in range(1, len(classes) + 1):
					# 		if K not in rank_acc.keys():
					# 			rank_acc[K] = 0
					# 		t = np.argpartition(pre[i], -K)[-K:]
					# 		if np.argmax(y_batch[i]) in t:
					# 			rank_acc[K] += 1
					# correct_num += acc * batch_size
					# total_num += batch_size
					# print(
					# 	'Testing Bacth: {:>3} - Validation Loss: {:>6.3f} - Validation Rank-1 Accuracy {:>6.3f}'
					# 		.format(cnt,
					# 	            loss,
					# 	            acc,
					# 	            ))
				# total_acc = correct_num / total_num
				print('Rank-1 Accuracy: %f' % total_acc)
				cal_AUC(score_y=preds, pred_y=ys, ps='nAUC')
			else:
				all_frame_preds = []
				for batch_i, (y_batch, X_batch) in enumerate(
						get_new_train_batches(y_2, X_2, batch_size)):
					loss, acc, pre = sess.run([cost, accuracy, pred],
					                          {X_input: X_batch,
					                           y_input: y_batch,
					                           lr: learning_rate})
					ys.extend(y_batch.tolist())
					preds.extend(pre.tolist())
					accs.append(acc)
					all_frame_preds.extend(pre)
					cnt += 1
					# for i in range(y_batch.shape[0]):
					# 	for K in range(1, len(classes) + 1):
					# 		if K not in rank_acc.keys():
					# 			rank_acc[K] = 0
					# 		t = np.argpartition(pre[i], -K)[-K:]
					# 		if np.argmax(y_batch[i]) in t:
					# 			rank_acc[K] += 1
					# correct_num += acc * batch_size
					# total_num += batch_size
					# print(
					# 	'Testing Bacth: {:>3} - Validation Loss: {:>6.3f} - Validation Rank-1 Accuracy {:>6.3f}'
					# 		.format(cnt,
					# 	            loss,
					# 	            acc,
					# 	            ))
				sequence_pred_correct = 0
				sequence_num = 0
				sequence_preds = []
				sequence_ys = []
				for k in range(len(all_frame_preds) // time_steps):
					sequence_labels = np.argmax(y_2[k * time_steps: (k + 1) * time_steps], axis=1)
					if (sequence_labels == np.tile(sequence_labels[0], [sequence_labels.shape[0]])).all():
						frame_predictions = np.array(all_frame_preds[k * time_steps: (k + 1) * time_steps])
						sequence_pred = np.argmax(np.average(frame_predictions, axis=0))
						if sequence_pred == sequence_labels[0]:
							sequence_pred_correct += 1
						sequence_num += 1
						sequence_ys.append(sequence_labels[0])
						aver = np.average(frame_predictions, axis=0)
						sequence_preds.append(aver)
				seq_acc_t = sequence_pred_correct / sequence_num
				# total_acc = correct_num / total_num
				# print('(Frame) Rank-1 Accuracy: %f' % total_acc)
				print('Rank-1 Accuracy: %f' % seq_acc_t)
				sequence_ys = label_binarize(sequence_ys, classes=classes)
				# cal_AUC(score_y=preds, pred_y=ys, ps='nAUC')
				cal_AUC(score_y=sequence_preds, pred_y=sequence_ys, ps='nAUC')

if __name__ == '__main__':
	tf.app.run()