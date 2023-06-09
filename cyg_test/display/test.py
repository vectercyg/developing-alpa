import jax
import pydot
from jax.core import Jaxpr,DropVar

import alpa


def test(x,y):
    a=x*y
    b=a+1
    c=a+2
    # return b

test_jaxpr=jax.make_jaxpr(test)(2,3)


drop_num=1
node_id_to_name=dict()
node_id_to_label=dict()
def CreateNode(var,shape="ellipse",color="blue"):
    global drop_num
    if(isinstance(var,DropVar)):
        name="dropvar"+var.__repr__()+str(drop_num)
        drop_num+=1
    else:
        name=var.__repr__()
        
    label=name+" "+var.aval.__repr__()
    node_id_to_name[id(var)]=name
    node_id_to_label[id(var)]=label
    return pydot.Node(name,label=label,shape=shape,color=color)

def CreateEdge(src_var,dst_var,label=None,color="blue"):
    return pydot.Edge(node_id_to_name[id(src_var)],node_id_to_name[id(dst_var)],label=label,color=color)

def IsExistedNode(var):
    if(node_id_to_name.get(id(var),None)!=None):
        return True
    else:
        return False


def JaxprToDot(jaxpr:Jaxpr,graph_name='G',graph_type='digraph', strict=False,suppress_disconnected=False, simplify=False, **attrs):
    graph = pydot.Dot(graph_name,graph_type=graph_type, strict=strict,suppress_disconnected=suppress_disconnected, simplify=simplify, **attrs)

    for var in jaxpr.invars:
        if(not IsExistedNode(var)):
            graph.add_node(CreateNode(var))
    for var in jaxpr.outvars:
        if(not IsExistedNode(var)):
            graph.add_node(CreateNode(var))
    for eqn in jaxpr.eqns:
        for var in eqn.invars:
            if(not IsExistedNode(var)):
                graph.add_node(CreateNode(var))
        for var in eqn.outvars:
            if(not IsExistedNode(var)):
                graph.add_node(CreateNode(var))
        for src_var in eqn.invars:
            for dst_var in eqn.outvars:
                graph.add_edge(CreateEdge(src_var=src_var,dst_var=dst_var,label=eqn.primitive.name))
    
    return graph

graph= JaxprToDot(test_jaxpr.jaxpr)
print(graph.to_string())
# graph.write
graph.write("cyg_test/display/output.png",format='png')