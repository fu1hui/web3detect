import json
import os


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
                    res_dict['fcnum'] = fc_num
                    lar_increase = json_largest_increase(transaction_data)
                    res_dict['max'] = lar_increase
                    lar_decrease = json_largest_decrease(transaction_data)
                    res_dict['min'] = lar_decrease
                    deepest_call = json_deepest_call(transaction_data)
                    res_dict['depth'] = deepest_call

                    save_json(res_dict, folder_name, filename)


def load_json(jpath):
    with open(jpath, 'r') as jf:
        jdata = json.load(jf)
        return jdata


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
        json.dump(data, json_file, indent=4, ensure_ascii=False)


'''yzk start'''


def json_largest_increase(jdata):  #extract the largest increase in amount
    changed_accounts_list = jdata["balance_change"]
    if not "balanceChanges" in changed_accounts_list:
        return -1
    changed_accounts_list = changed_accounts_list["balanceChanges"]
    laincre = -1  # largest increase
    for account in changed_accounts_list:
        assets = account["assets"]
        for change in assets:
            if change["sign"] == True and float(change["amount"].replace(',', '')) > laincre:
                laincre = float(change["amount"].replace(',', ''))
    return laincre


def json_largest_decrease(jdata):  #extract the largest decrease in amount
    lade = -1  #larget decrease
    changed_accounts_list = jdata["balance_change"]
    if not "balanceChanges" in changed_accounts_list:
        return -1
    changed_accounts_list = changed_accounts_list["balanceChanges"]
    for account in changed_accounts_list:
        assets = account["assets"]
        for change in assets:
            if change["sign"] == False and float(change["amount"].replace(',', '')) > lade:
                lade = float(change["amount"].replace(',', ''))
    return lade


def json_deepest_call(jdata):  #extract the largest depth of func call
    try:
        gasflame = (jdata["trace"])["gasFlame"]  #root
    except:
        return -1
    root = TreeNode(gasflame[0])
    return root.get_deepest()


class TreeNode:
    def __init__(self, node):
        self.depth = node["depth"]
        self.children = node["children"]  #type - list

    def isleaf(self):
        if len(self.children) == 0:
            return True
        return False

    def get_deepest(self):
        if self.isleaf():
            return self.depth
        return max((TreeNode(child).get_deepest() for child in self.children))


'''yzk end'''

if __name__ == '__main__':
    attack_data_path = './attack_data'
    normal_data_path = './normal_data'
    load_data_from_folder(attack_data_path, 'attack_json')
    load_data_from_folder(normal_data_path, 'normal_json')
