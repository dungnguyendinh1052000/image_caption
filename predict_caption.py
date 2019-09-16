# Thêm thư viện
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

########################### I. PHẦN ĐỌC CÁC THÔNG TIN MÔ TẢ ##################################
path_to_desc = "/home/dung/Work/image_caption/Flickr8k_text/Flickr8k.token.txt"


# Đọc file các caption
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


doc = load_doc(path_to_desc)

# Lưu caption dưới dạng key value: id_image : ['caption 1', 'caption 2', 'caption 3',' caption 4', 'caption 5']


def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# extract filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping


descriptions = load_descriptions(doc)

# Hàm tiền xử lý các câu mô tả như: đưa về chữ thường, bỏ dấu, bỏ các chữ số....


def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word) > 1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] = ' '.join(desc)


clean_descriptions(descriptions)

# Lưu description xuống file


def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


save_descriptions(
    descriptions, '/home/dung/Work/image_caption/descriptions.txt')

################################ II. PHẦN ĐỌC ẢNH  ẢNH ############################

path_to_train_id_file = '/home/dung/Work/image_caption/Flickr8k_text/Flickr_8k.trainImages.txt'
path_to_test_id_file = '/home/dung/Work/image_caption/Flickr8k_text/Flickr_8k.testImages.txt'
path_to_image_folder = '/home/dung/Work/image_caption/Flickr8k_Dataset/Flicker8k_Dataset/'


# Lấy id ảnh tương ứng với dữ liệu train, test, dev
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)


train_id = load_set(path_to_train_id_file)


# 1. -------- Xử lý ảnh dùng để train -----------------------------------------------


# Lấy lấy các ảnh jpg trong thư mục chứa toàn bộ ảnh Flickr
all_train_images_in_folder = glob.glob(path_to_image_folder + '*.jpg')


# Đọc toàn bộ nội dung file danh sách các file ảnh dùng để train
train_images = set(open(path_to_train_id_file, 'r').read().strip().split('\n'))

# Danh sách ấy sẽ lưu full path vào biến train_img
train_img = []

for image in all_train_images_in_folder:  # Duyệt qua tất cả các file trong folder
    # Nếu tên file của nó thuộc training set
    if image.split('/')[-1] in train_images:
        train_img.append(image)  # Thì thêm vào danh sách ảnh sẽ dùng để train

# 2. -------- Xử lý ảnh dùng để test (xử lý tương tự) -------------------------------------

test_images = set(open(path_to_train_id_file, 'r').read().strip().split('\n'))
test_img = []

for image in all_train_images_in_folder:
    if image.split('/')[-1] in test_images:
        test_img.append(image)

############################ III. PHẦN XỬ LÝ DỮ LIỆU MÔ TẢ ############################

# Hàm đọc mô tả từ file  và Thêm 'startseq', 'endseq' cho chuỗi


def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions


train_descriptions = load_clean_descriptions(
    '/home/dung/Work/image_caption/descriptions.txt', train_id)

# Tạo list các training caption
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)

# Chỉ lấy các từ xuất hiện trên 10 lần
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

# Tạo từ điển map từ Word sang Index và ngược lại
ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

# Tính toán bảng từ vựng
vocab_size = len(ixtoword) + 1  # Thêm 1 cho từ dùng để padding

# Chuyển thành từng dòng


def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# Tính toán độ dài nhất của câu mô tả


def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)


max_length = max_length(train_descriptions)

############################ IV. PHẦN XỬ LÝ ẢNH ĐẦU VÀO ĐỂ ĐƯA VAO MODEL CHÍNH  ############################

# Hàm load ảnh, resize về khích thước mà Inception v3 yêu cầu.


def preprocess(image_path):
    # Resize ảnh về 299x299 làm đầu vào model
    img = load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    # Thêm một chiều  nữa
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# Khởi tạo Model INception v3 để tạo ra feauture cho các ảnh của chúng ta
model1 = InceptionV3(weights='imagenet')
model_new = Model(model1.input, model1.layers[-2].output)

# Hàm biến ảnh đầu vào thành vector features (2048, )


def encode(image):
    images = preprocess(image)  # preprocess the image
    # Get the encoding vector for the image
    fea_vec = model_new.predict(images)
    # reshape from (1, 2048) to (2048, )
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

# Lặp một vòng qua các ảnh train và biến hết thành vector features


def train_features(image):
  img = '/home/dung/Work/image_caption/Flickr8k_Dataset/Flicker8k_Dataset/'+image
  encoding_train = encode(img)
  return encoding_train


def test_features(image):
  img = '/home/dung/Work/image_caption/Flickr8k_Dataset/Flicker8k_Dataset/'+image
  encoding_test = encode(img)
  return encoding_test
############################ V. PHẦN XỬ LÝ MÔ TẢ ĐẦU VÀO ĐỂ ĐƯA VAO MODEL CHÍNH  ############################


# Tải model Glove để embeding word
embeddings_index = {}  # empty dictionary
f = open('/home/dung/Work/image_caption/glove.6B.200d.txt', encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 200
# Tạo ma trận embeding cho bảng từ vững, mỗi từ embeding bằng 1 vector 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Lặp qua các từ trong danh sách từ
for word, i in wordtoix.items():
    # Lấy embeding của Glove gán vào embeding vector
    embedding_vector = embeddings_index.get(word)
    # Nếu như không None thì gán vào mảng Maxtrix
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


############################ VI. TẠO MODEL CHÍNH VÀ TIẾN HÀNH TRAIN  ############################


# image feature extractor model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# partial caption sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# decoder (feed forward) model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# merge the two input models
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
# Layer 2 dùng GLOVE Model nên set weight thẳng và không cần train
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.load_weights('/home/dung/Work/image_caption/model_30.h5')
model.summary()

# # Compile model
# model.compile(loss='categorical_crossentropy', optimizer='adam')


############################ VII. KIỂM THỬ XEM MODEL HOẠT ĐỘNG NHƯ NÀO  ############################


# Hàm đặt Caption
# Với môi ảnh mới khi test, ta sẽ bắt đầu chuỗi với 'startseq' rồi sau đó cho vào model để dự đoán từ tiếp theo. Ta thêm từ
# vừa được dự đoán vào chuỗi và tiếp tục cho đến khi gặp 'endseq' là kết thúc hoặc cho đến khi chuỗi dài 34 từ.
def setCaption(photo):
    in_text='starteseq'
    
    for i in range(max_length):
        sequence=[wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence=pad_sequences([sequence],maxlen=max_length)
        
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
image_name="Tay_dua_Le_Khanh_Loc_ub150_ARRC.jpg"
x=plt.imread(path_to_image_folder + image_name)

image_predict=train_features(image_name).reshape((1,2048))

# In kết quả mô tả do model tự đặt cho ảnh
print(setCaption(image_predict))
