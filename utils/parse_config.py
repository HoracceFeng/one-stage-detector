def parse_model_cfg(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_cfg(path):
    """Parses the data configuration file, has been deprecated. The .data file is replaced by config.json"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def parse_json_cfg(path):
    """Parses config.json"""
    import json

    def _modify_recursive_item(dictionary, attrs):
        """turn string to bool"""
        for item in dictionary:     
            if type(dictionary.get(item)) is dict:
                attrs.append(item)
                _modify_recursive_item(dictionary.get(item), attrs)
            else:
                if type(dictionary.get(item)) is str:
                    if dictionary[item] in ['True', 'true', 't', 'T']:
                        dictionary[item] = True
                    if dictionary[item] in ['False', 'false', 'f', 'F']:
                        dictionary[item] = False
                else:
                    continue

    with open(path) as config_buffer:
        config = json.loads(config_buffer.read())
    _modify_recursive_item(config, [])

    return config


def parse_dict_file(path):
    """load dictionary"""
    dictfile = open(path).readlines()
    classes = []
    for line in dictfile:
        cate = line.strip()
        if cate not in classes:
            classes.append(cate)
    return classes



if __name__ == '__main__':

    # options = parse_data_cfg('/code/data/sign/FT-prohibit.data')
    # print(options)

    config = parse_json_cfg('/code/cfg/config.json')
    print(config)
