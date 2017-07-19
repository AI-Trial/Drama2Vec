from prettyprint import pp
import numpy as np
import pandas
import tensorflow as tf
import math
from matplotlib import pylab
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

if __name__ == '__main__':
	import argparse
	cmd_parser = argparse.ArgumentParser(description='config for batch, embedding')
	cmd_parser.add_argument('--batch', metavar='batch_size', type=int, default=10, help='batch size')
	cmd_parser.add_argument('--embed', metavar='embedding_size', type=int, default=75, help='embedding size')
	cmd_parser.add_argument('--save_path', metavar='savepath', type=str, default='./CBOWresult/b10e75/', help='save path')
	cmd_parser.add_argument('--mode', metavar='mode', type=str, default='similarity', help='training/similarity/arithmetic/2d-visualize/2d-arithmetic/validation')
	cmd_parser.add_argument('--valid_id', metavar='valid_id', type=int, default=297, help='validation id for similality')
	cmd_parser.add_argument('--arith_list', metavar='[base, minus, plus]', nargs='+', type=int, default=[297,341,70], help='calcurate arithmetic relativity')
	cmd_parser.add_argument('--train_model', metavar='train_model', type=str, default='CBOW', help='CBOW or skipgram')
	cmd_parser.add_argument('--valid_list', metavar='4 arguments', nargs='+', type=int, default=[297,150,7,228], help='valid_id lists')

	args = cmd_parser.parse_args()
	row=678
	col=5
	max_epoch = 1000
	batch_size = args.batch
	embedding_size = args.embed
	save_path = args.save_path
	mode = args.mode
	valid_id = args.valid_id
	arith_list = args.arith_list
	train_model = args.train_model
	valid_list = args.valid_list

	"""
	data_operation
	#df:setlist for each drama -> use SKIPGRAM
	#data_dic:data and datacount
	#dic_onehot:data and onehot number
	#dic_category:data and category
	"""

	df = pandas.read_csv('./data_for_calc.csv', delimiter='\t', header=-1, encoding = 'utf8')

	data_dic ={}
	for r in np.arange(row):
		for c in np.arange(col):
			#print type(df.iloc[r,c])
			data_dic[(df.iloc[r,c])]=0
	for r in np.arange(row):
		for c in np.arange(col):
			data_dic[(df.iloc[r,c])]+=1
	data_dic['otherCast']=0
	data_dic['otherSinger']=0
	for r in np.arange(row):
		for c in np.arange(col):
			if data_dic[(df.iloc[r,c])] == 1:
				if c==1 or c==2:
					del data_dic[(df.iloc[r,c])]
					data_dic['otherCast']+=1
					df.iloc[r,c]='otherCast'
				elif c==3:
					del data_dic[(df.iloc[r,c])]
					data_dic['otherSinger']+=1
					df.iloc[r,c]='otherSinger'			
	#print len(data_dic)
	dic_category = {}
	for r in np.arange(row):
		for c in np.arange(col):
			dic_category[(df.iloc[r,c])] = c

	dic_onehot = {}
	num = 0
	for key in data_dic.keys():
		dic_onehot[key]=num
		num=num+1
	#print 'amount of data:', len(dic_onehot)
	data_size = len(dic_onehot)

	#description of mode
	if mode == 'training':
		print '--training as ' + train_model + '--'
	else:
		print '--evaluating as ' + mode + '--'

	"""
	network model
	"""

	#CBOW
	train_data = tf.placeholder(tf.float32, [batch_size, col-1, data_size])
	labels = tf.placeholder(tf.float32, [batch_size, data_size])
	valid_data = tf.placeholder(tf.float32, [1, col-1, data_size])
	valid_labels = tf.placeholder(tf.float32, [1, data_size])

	embeddings = tf.Variable(
	    tf.random_uniform([data_size, embedding_size], -1.0, 1.0))
	weights = tf.Variable(
	    tf.truncated_normal([embedding_size, data_size],
	                         stddev=1.0 / math.sqrt(embedding_size)))
	biases = tf.Variable(tf.zeros([data_size]))

	if train_model == 'CBOW':
		for i in np.arange(col-1):
			embed = tf.matmul(train_data[:,i,:], embeddings)
			if i == 0:
				embed_mean = embed
			else:
				embed_mean = tf.add(embed_mean, embed)
		embed_mean = embed_mean/(col-1)
		logits = tf.add(tf.matmul(embed_mean, weights), biases)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

		optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

		saver = tf.train.Saver()

		for i in np.arange(col-1):
			embed = tf.matmul(valid_data[:,i,:], embeddings)
			if i == 0:
				embed_mean = embed
			else:
				embed_mean = tf.add(embed_mean, embed)
		embed_mean = embed_mean/(col-1)
		logits_valid = tf.add(tf.matmul(embed_mean, weights), biases)
		logits_valid = tf.nn.softmax(logits_valid)

	#skipgram
	if train_model == 'skipgram':
		embed = tf.matmul(labels, embeddings)
		logits = tf.add(tf.matmul(embed, weights), biases)
		for i in np.arange(col-1):
			label=train_data[:,i,:]
			if i == 0:
				loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =logits, labels=label))
			else:
				loss = tf.add(loss, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =logits, labels=label)))
		loss =loss/(col-1)

		optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

		saver = tf.train.Saver()

		embed = tf.matmul(valid_labels, embeddings)
		logits_valid = tf.add(tf.matmul(embed, weights), biases)
		logits_valid = tf.nn.softmax(logits_valid)


	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm

	extract_data = tf.placeholder(tf.float32, [1, data_size])
	extract_embed = tf.matmul(extract_data, normalized_embeddings)
	sim_data = tf.placeholder(tf.float32, [1, embedding_size])
	normalized_sim_data = sim_data / tf.sqrt(tf.reduce_sum(tf.square(sim_data), 1, keep_dims=True))
	similality = tf.matmul(normalized_sim_data, tf.transpose(normalized_embeddings))

	if mode == 'training':
		"""
		Training
		"""
		def next_batch(dataset_num, order, batch_size):
			epoch_flag = True
			if order + batch_size > row*col:
				order = 0
				epoch_flag = False
			if order == 0:
				np.random.shuffle(dataset_num)
			batch_order = dataset_num[order:order+batch_size]
			order = order + batch_size
			return batch_order, dataset_num, order, epoch_flag

		ite = 0
		dataset_num = np.arange(row*col)
		order=0
		record_loss = []
		ite_epoch = row*col // batch_size

		sess = tf.Session()

		tf.initialize_all_variables().run(session=sess)
		for epoch in np.arange(max_epoch):
			epoch_flag = True
			stack_loss = 0
			while epoch_flag:
				batch_order, dataset_num, order, epoch_flag = next_batch(dataset_num, order, batch_size)
				#print batch_order
				for (i,x) in enumerate(batch_order):
					input_key = df.iloc[x//col, x%col]
					input_num = dic_onehot[input_key]
					input_vec = np.zeros([1, data_size])
					input_vec[0,input_num] = 1
					output_vec = np.zeros([1, col-1, data_size])
					num = 0
					for c in np.arange(col):
						if c != x%col:
							output_key = df.iloc[x//col, c]
							output_num = dic_onehot[output_key]
							output_vec[0,num,output_num]=1
							num+=1
					if i == 0:
						input_vecs = input_vec
						output_vecs = output_vec
					else:
						input_vecs = np.concatenate((input_vecs,input_vec), axis=0)
						output_vecs = np.concatenate((output_vecs, output_vec), axis=0)

				_, loss_val = sess.run([optimizer, loss], feed_dict={
					train_data:output_vecs,
					labels:input_vecs
					})

				print 'epoch:', epoch, '  iteration:', ite, '  loss:', loss_val

				stack_loss += loss_val
				ite +=1

			mean_loss = stack_loss/ite_epoch
			record_loss.append([epoch+1, mean_loss])

		save_path=saver.save(sess, save_path + 'checkpoint_')
		print record_loss
		np.savetxt(save_path + 'record_loss.csv', record_loss, delimiter=',')


	else:
		"""
		evaluation
		"""
		sess = tf.Session()
		saver.restore(sess, save_path + 'checkpoint_')
		final_embeddings = normalized_embeddings.eval(session=sess)

		def get_embed(id_):
			input_vec = np.zeros([1, data_size])
			input_vec[0,id_] = 1
			extract_embed_val = sess.run(extract_embed, feed_dict={extract_data:input_vec})
			return extract_embed_val

		if mode == 'similarity':
			#similality
			valid_embed = get_embed(valid_id)
			sim = sess.run(similality, feed_dict={sim_data:valid_embed})
			nearest=(-sim).argsort()
			for i in np.arange(8):
				for key, value in dic_onehot.items():
					if value == nearest[0,i]:
						if i == 0:
							print_str = 'Nearest to ' + key
							pp(print_str)
							print '----'
						else:
							print_str = key + '    cosine value :' + str(sim[0, value])
							pp(print_str)
						break
		
		
		if mode == 'arithmetic':
			#arithmetic
			#print arith_list
			arith_embed_1 = get_embed(arith_list[0])
			arith_embed_2 = get_embed(arith_list[1])
			arith_embed_3 = get_embed(arith_list[2])
			arith_embed = np.add(np.subtract(arith_embed_1, arith_embed_2), arith_embed_3)
			sim = sess.run(similality, feed_dict={sim_data:arith_embed})
			nearest=(-sim).argsort()
			arith_list_key = []
			for i in arith_list:
				for key, value in dic_onehot.items():
					if value == i:
						arith_list_key.append(key)
			print_str = '[' + arith_list_key[0] + '] minus [' + arith_list_key[1] + '] plus [' + arith_list_key[2] + ']'
			pp(print_str)
			print '----'
			for i in np.arange(8):
				for key, value in dic_onehot.items():
					if value == nearest[0,i]:
						print_str = key + '    cosine value :' + str(sim[0, value])
						pp(print_str)
						break


		if mode == '2d-visualize' or '2d-arithmetic':
			#2D-representation

			tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000000)
			svd = TruncatedSVD(n_components=2, random_state=0)
			two_d_embeddings = svd.fit_transform(final_embeddings[:, :])
			#two_d_embeddings = tsne.fit_transform(final_embeddings[:, :])

			def plot(embeddings, labels, category):
				assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
				pylab.figure(figsize=(10,10))  # in inches
				for i, label in enumerate(labels):
					x, y = embeddings[i,:]
					cat = category[i]
					pylab.scatter(x, y, color=col(cat))
					pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
			                   ha='right', va='bottom', fontsize=7)
				pylab.show()

			def col(category):
				if category == 0:
					col = 'b'
				elif category == 1:
					col = 'g'
				elif category == 2:
					col = 'g'
				elif category == 3:
					col = 'm'
				elif category == 4:
					col = 'r'
				return col

			words = []
			category = []
			for i in np.arange(data_size):
				for key, value in dic_onehot.items():
					if value == i:
						words.append(key)
						category.append(dic_category[key])
						break

			if mode == '2d-visualize':
				plot(two_d_embeddings, words, category)

			elif mode == '2d-arithmetic':
				arith_embed_1 = two_d_embeddings[arith_list[0]]
				arith_embed_2 = two_d_embeddings[arith_list[1]]
				arith_embed_3 = two_d_embeddings[arith_list[2]]
				arith_embed = np.add(np.subtract(arith_embed_1, arith_embed_2), arith_embed_3)
				sim = np.zeros([data_size])
				
				for i in np.arange(data_size):
					sim[i] = ((arith_embed[0] - two_d_embeddings[i,0])**2 + (arith_embed[1] - two_d_embeddings[i,1])**2)**0.5

				nearest=(sim).argsort()
				
				arith_list_key = []
				for i in arith_list:
					for key, value in dic_onehot.items():
						if value == i:
							arith_list_key.append(key)
				print_str = '[' + arith_list_key[0] + '] minus [' + arith_list_key[1] + '] plus [' + arith_list_key[2] + ']'
				pp(print_str)
				print '----'
				for i in np.arange(8):
					for key, value in dic_onehot.items():
						if value == nearest[i]:
							pp(key)
							break

		if mode == 'validation':
			num = 0
			output_vec = np.zeros([1, 4, data_size])
			for i in valid_list:
				output_vec[0,num,i]=1
				num +=1

			valid_list_key = []
			for i in valid_list:
				for key, value in dic_onehot.items():
					if value == i:
						valid_list_key.append(key)
						break

			logits_valid_val = sess.run(
				logits_valid,
				feed_dict={valid_data:output_vec})
			#print logits_valid_val.shape
			#print np.argmax(logits_valid_val)
			predict = (-logits_valid_val).argsort()

			print 'input'
			pp(valid_list_key)

			print 'prediction'
			for i in np.arange(3):
					for key, value in dic_onehot.items():
						if value == predict[0,i]:
							print_str = key + '   confidence :' + str(logits_valid_val[0,value])
							pp(print_str)
							break



				
				



