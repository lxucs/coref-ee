import pickle
import json


f = open('/home/lxu85/data_dir/dev.english.128.jsonlines', 'rb')
data_list = []
for line in f.readlines():
    data = json.loads(line)
    data_list.append(data)
a = 1
