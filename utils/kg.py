import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
bar_format = '{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'


def get_ent_rel(path_data):
    with open(path_data + 'entities.txt', 'r') as f:
        ent_list = f.read().split('\n')
    with open(path_data + 'relations.txt', 'r') as f:
        rel_list = f.read().split('\n')
    ent_list = ent_list[:-1] if ent_list[-1] == '' else ent_list
    rel_list = rel_list[:-1] if rel_list[-1] == '' else rel_list
    return ent_list, rel_list


def get_facts(path_data):
    facts = list()
    for data in ['test', 'dev', 'train']:
        file = path_data + f'{data}.txt'
        with open(file, 'r') as f:
            facts_f = f.read().split('\n')
        facts_f = facts_f[:-1] if facts_f[-1] == '' else facts_f
        facts += facts_f
    facts = [fact.split('\t') for fact in facts]
    facts = [fact[:3] for fact in facts if len(fact) == 3 or fact[-1] == '1']
    return facts


def get_graph(path_data, entity_list, relation_list):
    facts = get_facts(path_data)

    nodes = {
        entity: {'out':  dict(), 'in': dict()} for entity in entity_list
    }
    edges = {
        relation: {'in':  list(), 'out': list()} for relation in relation_list
    }
    for fact in tqdm(facts, bar_format=bar_format):
        subject, relation, object = fact
        node_s = nodes[subject]
        node_s_out = node_s['out'].get(relation, list()) + [object]
        node_s['out'][relation] = node_s_out
        node_o = nodes[object]
        node_o_in = node_o['in'].get(relation, list()) + [subject]
        node_o['in'][relation] = node_o_in
        nodes[subject] = node_s
        nodes[object] = node_o

        edge_r = edges[relation]
        edge_out = edge_r['out']
        edge_out.append(object)
        edge_r['out'] = edge_out
        edge_in = edge_r['in']
        edge_in.append(subject)
        edge_r['in'] = edge_in
        edges[relation] = edge_r

    for entity, node in nodes.items():
        for relation, objects in node['out'].items():
            node['out'][relation] = list(set(objects))
        for relation, objects in node['in'].items():
            node['in'][relation] = list(set(objects))
        nodes[entity] = node

    for relation, edge in edges.items():
        edge['out'] = list(set(edge['out']))
        edge['in'] = list(set(edge['in']))
        edges[relation] = edge

    return nodes, edges


def get_positives(fact, nodes):
    s, R, o = fact
    r_list = R.split(',')
    entities = [s]
    for r in r_list:
        if r[:2] == '**':
            rel = r[2:]
            edge = 'in'
        else:
            rel = r
            edge = 'out'
        ent_next = list()
        for node in (nodes[ent] for ent in entities):
            ent_next += node[edge].get(rel, list())
        entities = list(set(ent_next))
    return entities


def get_negatives(fact, edges):
    s, R, o = fact
    r_list = R.split(',')
    r = r_list[-1]
    if r[:2] == '**':
        r_last = r[2:]
        edge = 'in'
    else:
        r_last = r
        edge = 'out'
    return edges[r_last][edge]


# def get_dataset(data, entity_dict, relation_dict, path_data):
#     with open(path_data + f'{data}.txt', 'r') as f:
#         lines = f.read().split('\n')
#     lines = lines[:-1] if lines[-1] == '' else lines
#     facts = [line.split('\t') for line in lines]
#     facts = [fact[:3] for fact in facts if len(fact) == 3 or fact[-1] == '1']
#
#     idx_list, s_list, R_list, o_list = list(), list(), list(), list()
#     for i, fact in enumerate(facts):
#         s, R, o = fact
#         idx_list.append(i)
#         s_list.append(entity_dict[s])
#         r_list = R.split(',')
#         R_list.append([relation_dict[r] for r in r_list])
#         o_list.append(entity_dict[o])
#
#     data = TensorDataset(
#         *[torch.LongTensor(L) for L in [idx_list, s_list, R_list, o_list]]
#     )
#     return data, facts


def get_path_dataset(data, size, entity_dict, relation_dict_both, path_data):
    with open(path_data + f'paths/{data}', 'r') as f:
        lines = f.read().split('\n')
    lines = lines[:-1] if lines[-1] == '' else lines
    facts = [line.split('\t') for line in lines]
    facts_out = list()
    idx_list, s_list, R_list, o_list = list(), list(), list(), list()
    i = 0
    for fact in facts:
        s, R, o = fact
        r_list = R.split(',')
        if len(r_list) != size:
            continue
        facts_out.append(fact)
        idx_list.append(i)
        s_list.append(entity_dict[s])
        R_list.append([relation_dict_both[r] for r in r_list])
        o_list.append(entity_dict[o])
        i += 1
    data = TensorDataset(
        *[torch.LongTensor(L) for L in [idx_list, s_list, R_list, o_list]]
    )
    return data, facts_out
