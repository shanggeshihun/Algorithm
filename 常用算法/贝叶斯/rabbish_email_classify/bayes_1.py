import os 

def readtxt(path,encoding):
	with open(path,'r',encoding=encoding) as f:
		lines=f.readlines()
	return lines 
	
def fileWalker(path):
	"""
	path:路径
	return:返回该路径下的所有文件路径列表
	"""
	fileArray=[]
	for root,dirs,files in os.walk):
		for fn in files:
			eachpath=str(root+'\\'+fn)
			fileArray.append(eachpath)
	return fileArray
	
def email_parser(email_path):
	"""
	email_path:邮件文件的路径
	return:邮件分词
	"""
	punctutations=""",.<>()*&^%$#@!'";~`[]{}|、\\/~+_-=?"""
	content_list=readtxt(email_path,'utf-8')
	content=(' '.join(content_list)).replace('\r\n',' ').replace('\t',' ')
	clean_word=[]
	for punctutation in punctutations:
		content=(' '.join(content.split(punctutation))).replace('  ',' ')
		clean_word=[word.lower() for word in content.split(' ') if len(word)>2]
	return clean_word 

def get_word(email_file):
	"""
	email_file:邮件文件夹路径
	return:所有邮件分词列表，分词集合
	"""
	word_list=[]
	word_set=[]
	email_paths=fileWalker(email_file)
	for email_path in email_paths:
		clean_word=email_parser(email_path)
		word_list.append(clean_word)
		word_set.extend(clean_word)
	return word_list,set(word_set)
	

def count_word_prob(email_list,union_set):
	word_prob={}
	for word in union_set:
		counter=0
		for email in email_list：
			if word in email:
				counter+=1
			else:
				continue
		prob=0.0
		if counter!=0:
			prob=counter/len(email_list)
		else:
			prob=0.01
		word_prob[word]=prob
	return word_prob
	
def filter(ham_word_pro,spam_word_pro,test_file):
	test_paths=fileWalker(test_path)
	for test_path in test_paths:
		email_spam_prob=0.0
		spam_prob=0.5
		ham_prob=0.5
		file_name=test_path.split('\\')[-1]
		prob_dict={}
		words=set(email_parser(test_path))
		for word in words:
			Psw=0.0
			if word not in spam_word_pro:
				Psw=0.4
			else:
				Pws=spam_word_pro[word]
				Pwh=ham_word_pro[word]
				Psw=spam_prob*(Pws/(Pwh*ham_prob+Pws*spam_prob))
			prob_dict[word]=Psw 
		numerator=1
		denominar_h=1
		for k,v in prob_dict.items():
			numerate*=v 
			denominar_h*=(1-v)
		email_spam_prob=round(numerator+denominar_h),4)
		if email_spam_prob>0.5:
			print(file_name,'spam',email_spam_prob)
		else:
			print(file_name,'ham',email_spam_prob)


def main():
    ham_file = r'..\email\ham'
    spam_file = r'..\email\spam'
    test_file = r'..\email\test'
    ham_list,ham_set = get_word(ham_file)
    spam_list,spam_set = get_word(spam_file)
    union_set = ham_set | spam_set
    ham_word_pro = count_word_prob(ham_list, union_set)
    spam_word_pro = count_word_prob(spam_list, union_set)
    filter(ham_word_pro, spam_word_pro, test_file)

if __name__ == '__main__':
    main()
