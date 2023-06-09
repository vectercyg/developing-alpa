from typing import Dict
from ios.ir import *
import pydot
import json
#https://graphviz.gitlab.io/documentation/

#change colors ref:https://graphviz.org/doc/info/colors.html
# change nodes style:https://www.cnblogs.com/shuqin/p/11897207.html
#node shape:https://graphviz.gitlab.io/doc/info/shapes.html

colors = [
    "cadetblue1",
    "skyblue1",
    "royalblue",
    "aquamarine",
    "greenyellow",
    "teal",
    "gold",
    "green",
    "darkgreen",
    "darkorchid1",
    "hotpink",
    "khaki1",
    "indianred2",
    "sienna1"
]
cur_color = 0


def reset_color():
    global cur_color
    cur_color = 0


def get_next_color():
    global cur_color
    ret = colors[cur_color]
    cur_color = (cur_color + 1) % len(colors)
    return ret

def get_node_labelNostages(node):
    flops = node.flops() / 10 ** 6
    if isinstance(node, Placeholder):
        label = "\n".join([
            'Placeholder',
            f'{node.hint_name}#{flops:.1f}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Conv):
        label = "\n".join([
            'Conv',
            f'{node.hint_name}#{flops:.1f}',
            f'kernel {node.kernel}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Pool):
        label = "\n".join([
            'Pool',
            f'{node.hint_name}#{flops:.1f}',
            f'type {node.pool_type}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Identity):
        label = "\n".join([
            'Identity',
            f'{node.hint_name}#{flops:.1f}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Relu):
        label = "\n".join([
            'Relu',
            f'{node.hint_name}#{flops:.1f}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Element):
        label = "\n".join([
            node.op_type.capitalize(),
            f'{node.hint_name}#{flops:.1f}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Activation):
        label = "\n".join([
            node.act_type.capitalize(),
            f'{node.hint_name}#{flops:.1f}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Sequential):
        label = "\n".join([
            f'{node.hint_name}',
            f'{node.name}#{flops:.1f}',
            f'{node.output_shape}',
        ])
    else:
        raise ValueError
    return label

def get_node_label2(node, stage):
    if isinstance(node, Placeholder):
        label = "\n".join([
            'Placeholder',
            f'start oder {node.hint_name}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Conv):
        label = "\n".join([
            'Conv',
            f'start oder {node.hint_name}',
            f'kernel {node.kernel}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Pool):
        label = "\n".join([
            'Pool',
            f'start oder {node.hint_name}',
            f'type {node.pool_type}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Identity):
        label = "\n".join([
            'Identity',
            f'start oder {node.hint_name}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Relu):
        label = "\n".join([
            'Relu',
            f'start oder {node.hint_name}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Element):
        label = "\n".join([
            node.op_type.capitalize(),
            f'start oder {node.hint_name}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Activation):
        label = "\n".join([
            node.act_type.capitalize(),
            f'start oder {node.hint_name}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Sequential):
        label = "\n".join([
            f'start oder {node.hint_name}',
            f'{node.output_shape}',
        ])
    else:
        raise ValueError
    return label

def get_node_label(node, stage):
    flops = node.flops() / 10 ** 6
    if isinstance(node, Placeholder):
        label = "\n".join([
            'Placeholder',
            f'{node.hint_name} @ {stage} #{flops:.1f}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Conv):
        label = "\n".join([
            'Conv',
            f'{node.hint_name} @ {stage} #{flops:.1f}',
            f'kernel {node.kernel}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Pool):
        label = "\n".join([
            'Pool',
            f'{node.hint_name} @ {stage} #{flops:.1f}',
            f'type {node.pool_type}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Identity):
        label = "\n".join([
            'Identity',
            f'{node.hint_name} @ {stage} #{flops:.1f}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Relu):
        label = "\n".join([
            'Relu',
            f'{node.hint_name} @ {stage} #{flops:.1f}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Element):
        label = "\n".join([
            node.op_type.capitalize(),
            f'{node.hint_name} @ {stage} #{flops:.1f}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Activation):
        label = "\n".join([
            node.act_type.capitalize(),
            f'{node.hint_name} @ {stage} #{flops:.1f}',
            f'{node.output_shape}'
        ])
    elif isinstance(node, Sequential):
        label = "\n".join([
            f'{node.hint_name}',
            f'{node.name} @ {stage} #{flops:.1f}',
            f'{node.output_shape}',
        ])
    else:
        raise ValueError
    return label

# def drawJson(JsonFileName,filename:str):
#     dgraph=pydot.Dot(graph_type='digraph', label=label)
#     node_style = {
#         'shape': 'ellipse',
#         'style': '"rounded,filled"'
#     }
#     with open(JsonFileName, 'r') as Rjson_file:
#         readJson=json.load(Rjson_file)
#     node2dnode = {}
#     for nodes in readJson['nodes']:
#         label = "\n".join([nodes['name'],
#             f'id is {nodes['id']}',
#             f'mem{nodes['size']}',
#             f'time{nodes['fpgaLatency']}',
#         ])
#         dnode = pydot.Node(nodes['id'], label=label, **node_style, fillcolor=get_next_color())
#         node2dnode[node] = dnode
#         dgraph.add_node(dnode) 
#     for edges in readJson['edges']:
#         src=node2dnode[edges['sourceId']]
#         dst=node2dnode[edges['destId']]
#         e = pydot.Edge(src, dst, label=f'[{value.begin}:{value.end}] {index}')
#         dgraph.add_edge(e)
#     dgraph.write(fname, format='png')
def drawTopo(graph: Graph, fname: str, label: str = ""):
    dgraph = pydot.Dot(graph_type='digraph', label=label)

    node_style = {
        'shape': 'ellipse',
        'style': '"rounded,filled"'
    }

    name2stage = {}
    name2color = {}
    index = 0
    name2stage[graph.enter_node.name] = index
    reset_color()
    name2color[graph.enter_node.name] = get_next_color()
    index += 1
    for block in graph.blocks:
        for nodes in block.inner_nodes:
            color = get_next_color()
            index += 1
            name2stage[nodes.name] = index
            name2color[nodes.name] = color

    node2dnode = {}
    for node in graph.nodes():
        label = get_node_labelNostages(node)
        dnode = pydot.Node(node.name, label=label, **node_style, fillcolor=name2color[node.name])
        node2dnode[node] = dnode
        dgraph.add_node(dnode)
    for node in graph.nodes():
        for index, term in enumerate(node.inputs):
            for value in term:
                src = node2dnode[value.node]
                dst = node2dnode[node]
                e = pydot.Edge(src, dst, label=f'[{value.begin}:{value.end}] {index}')
                dgraph.add_edge(e)
    dgraph.write(fname, format='png')
def draw(graph: Graph, fname: str, label: str = ""):
    """
    Draw the computation graph and execution schedule.

    :param graph: ios.Graph
        The computation graph and schedule.

    :param fname:
        The output figure filename.

    :param label:
        The label of the figure.
    """
    dgraph = pydot.Dot(graph_type='digraph', label=label)

    node_style = {
        'shape': 'ellipse',
        'style': '"rounded,filled"'
    }

    name2stage = {}
    name2color = {}
    index = 0
    name2stage[graph.enter_node.name] = index
    reset_color()
    name2color[graph.enter_node.name] = get_next_color()
    index += 1
    for block in graph.blocks:
        for stage in block.stages:
            color = get_next_color()
            index += 1
            for seq in stage:
                for node_name in seq:
                    name2stage[node_name] = index
                    name2color[node_name] = color
    # print("finish dict")

    node2dnode = {}
    for node in graph.nodes():
        label = get_node_label2(node, name2stage[node.name])
        dnode = pydot.Node(node.name, label=label, **node_style, fillcolor=name2color[node.name])
        node2dnode[node] = dnode
        dgraph.add_node(dnode)
    # print("finish node")
    for node in graph.nodes():
        for index, term in enumerate(node.inputs):
            for value in term:
                src = node2dnode[value.node]
                dst = node2dnode[node]
                e = pydot.Edge(src, dst, label=f'[{value.begin}:{value.end}] {index}')
                dgraph.add_edge(e)
    # print("finish edges")
    dgraph.write(fname, format='png')
    epsName=fname[:-3]+'eps'
    dgraph.write(epsName, format='eps')


def draw_block(block: Block, fname, label: str = ""):
    dgraph = pydot.Dot(graph_type='digraph', label=label)

    node_style = {
        'shape': 'box',
        'style': '"rounded,filled"'
    }

    name2stage = {}
    name2color = {}
    index = 0
    name2stage[block.enter_node.name] = index
    index += 1
    reset_color()
    name2color[block.enter_node.name] = get_next_color()

    for stage in block.stages:
        color = get_next_color()
        for seq in stage:
            for node_name in seq:
                name2stage[node_name] = index
                name2color[node_name] = color
        index += 1

    node2dnode = {}
    for node in [block.enter_node] + block.inner_nodes + [block.exit_node]:
        label = get_node_label(node, name2stage[node.name])
        if node is block.exit_node:
            label += "\nexit_node"
        dnode = pydot.Node(node.name, label=label, **node_style, fillcolor=name2color[node.name])
        node2dnode[node] = dnode
        dgraph.add_node(dnode)
    for node in block.inner_nodes + [block.exit_node]:
        for index, term in enumerate(node.inputs):
            for value in term:
                src = node2dnode[value.node]
                dst = node2dnode[node]
                e = pydot.Edge(src, dst, label=f'[{value.begin}:{value.end}] {index}')
                dgraph.add_edge(e)
    dgraph.write(fname, format='png')


def draw_igraph(graph: Dict[int, List[int]], fname, label, chains: List[List[int]] = None):
    icolor = {}
    reset_color()
    if chains is None:
        for u in graph:
            icolor[u] = colors[cur_color]
    else:
        for chain in chains:
            for u in chain:
                icolor[u] = colors[cur_color]
            get_next_color()
    node_style = {
        'shape': 'box',
        'style': '"rounded,filled"'
    }
    dgraph = pydot.Dot(graph_type='digraph', label=label)
    idnode = {}
    for u in graph:
        idnode[u] = pydot.Node(str(u), **node_style, fillcolor=icolor[u])
        dgraph.add_node(idnode[u])
    for u in graph:
        for v in graph[u]:
            e = pydot.Edge(src=idnode[u], dst=idnode[v])
            dgraph.add_edge(e)
    dgraph.write(fname, format='png')
