import json
import os

attack_data_path = './attack_data'
normal_data_path = './normal_data'


#####################################
## 保存文件还没有写，先把你的数据处理写上 ##
#####################################

def load_data_from_folder(folder_path, folder_name):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                transaction_data = json.load(f)
                if 'trace' in transaction_data and 'dataMap' in transaction_data['trace']:
                    keys = ['gas', 'max', 'min', 'depth', 'fcnum']
                    res_dict = {key: None for key in keys}
                    gas = get_gas_value(transaction_data)
                    res_dict['gas'] = gas
                    fc_num = function_count(transaction_data)
                    res_dict['fc_num'] = fc_num
                    # 你的函数和字典赋值

                    save_json(res_dict, folder_name, filename)


def function_count(transaction_data):
    function_dic = {}

    data_map = transaction_data['trace']['dataMap']

    for key, value in data_map.items():
        node_type = value.get('nodeType')
        function = None

        if node_type == 0:
            invocation = value.get('invocation')
            if isinstance(invocation, dict):
                decoded_method = invocation.get('decodedMethod')
                if isinstance(decoded_method, dict):
                    function = decoded_method.get('name')
        elif node_type == 1:
            event = value.get('event')
            if isinstance(event, dict):
                decoded_log = event.get('decodedLog')
                if isinstance(decoded_log, dict):
                    function = decoded_log.get('name')

        if function:
            function_dic[function] = function_dic.get(function, 0) + 1

    max_value = max(function_dic.values(), default=-1)

    return max_value


def get_gas_value(transaction_data):
    profile = transaction_data.get('profile', {})
    basic_info = profile.get('basic_info', {})
    if 'gasUsed' in basic_info and basic_info['gasUsed']:
        return basic_info['gasUsed']
    return -1


def save_json(data, folder_name, json_filename):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    file_path = os.path.join(folder_name, json_filename)

    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)  # indent=4 格式化输出


load_data_from_folder(attack_data_path, 'attack_json')
load_data_from_folder(normal_data_path, 'normal_json')