from typing import Any, Dict, List, Optional, Sequence

import dill
import pydot
from jax._src import source_info_util
from jax.core import ClosedJaxpr, DropVar, Jaxpr, JaxprEqn, Literal, Primitive

from alpa.pipeline_parallel.computation import JaxPipelineComputation
from alpa.pipeline_parallel.primitive_def import pipeline_p

colors = [
    "red",
    "blue",
    "yellow",
    "purple",
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


drop_num=1
literal_num=1
node_id_to_name=dict()
node_id_to_label=dict()
def CreateNode(var,shape="ellipse",color="blue"):
    global drop_num,literal_num
    if(isinstance(var,DropVar)):
        name="dropvar"+var.__repr__()+str(drop_num)
        drop_num+=1
    if(isinstance(var,Literal)):
        name="literalvar_"+var.__repr__()+"_"+str(literal_num)
        literal_num+=1
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
    global cur_color
    for var in jaxpr.invars:
        if(not IsExistedNode(var)):
            graph.add_node(CreateNode(var,color=colors[cur_color]))
    for var in jaxpr.outvars:
        if(not IsExistedNode(var)):
            graph.add_node(CreateNode(var,color=colors[cur_color]))
    for eqn in jaxpr.eqns:
        if(eqn.primitive.name=="pipeline_marker" and eqn.params["mark_type"]=="start"):
            cur_color+=1
        for var in eqn.invars:
            if(not IsExistedNode(var)):
                graph.add_node(CreateNode(var,color=colors[cur_color]))
        for var in eqn.outvars:
            if(not IsExistedNode(var)):
                graph.add_node(CreateNode(var,color=colors[cur_color]))
        if(eqn.primitive.name=="pipeline_marker"):
            subgraph=pydot.Subgraph(rank="same")
            for i in range(len(eqn.invars)):
                subgraph.add_node(graph.get_node(node_id_to_name[id(eqn.outvars[i])])[0])
                label="{} {} {}".format(eqn.primitive.name,eqn.params["name"],eqn.params["mark_type"])
                graph.add_edge(CreateEdge(src_var=eqn.invars[i],dst_var=eqn.outvars[i],label=label))
            graph.add_subgraph(subgraph)
        else:
            for src_var in eqn.invars:
                for dst_var in eqn.outvars:
                    graph.add_edge(CreateEdge(src_var=src_var,dst_var=dst_var,label=eqn.primitive.name))
    
    return graph

def SliceComputeGradJaxprToForwardAndBackward(
        closed_jaxpr: ClosedJaxpr,layer_num:int) -> Sequence[JaxPipelineComputation]:
    """修改自slice_closed_jaxpr_by_full_pipeline_marks"""
    global_consts_dir = dict(
        zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))

    result_computations = []
    current_computation = None
    current_computation_name_list=["forward","backward"]
    layer_count=0
    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive.name =="pipeline_marker" and eqn.params["mark_type"] == "start":
            if(layer_count%layer_num==0):
                assert current_computation is None, (
                    "Defining a pipeline computation "
                    "inside a pipeline computation is "
                    "not allowed.")
                current_computation = JaxPipelineComputation(
                    name=current_computation_name_list[int(layer_count/layer_num)])
                for var in eqn.invars:
                    if isinstance(var, Literal):
                        pass
                    elif var in global_consts_dir:
                        current_computation.consts_dir[var] = global_consts_dir[var]
                    else:
                        current_computation.invars.append(var)
            layer_count+=1

        for var in eqn.invars:
            if not isinstance(var, Literal) and var in global_consts_dir:
                current_computation.consts_dir[var] = global_consts_dir[var]

        assert current_computation is not None
        current_computation.eqns.append(eqn)

        if eqn.primitive.name =="pipeline_marker" and eqn.params["mark_type"] == "end":
            if(layer_count%layer_num==0):
                assert current_computation is not None, (
                    "Ending a pipeline computation before its start.")
                # assert current_computation.name == eqn.params["name"], (
                #     "Ending a pipeline computation different from its start.")
                for var in eqn.outvars:
                    current_computation.outvars.append(var)
                result_computations.append(current_computation)
                current_computation = None
    assert len(result_computations) == 2,("前向与反向划分不正确")
    forward_closed_jaxpr = result_computations[0].closed_jaxpr()
    backward_closed_jaxpr = result_computations[1].closed_jaxpr()

    return forward_closed_jaxpr,backward_closed_jaxpr


def SerializeJaxpr(jaxpr:Jaxpr)->Dict:
    jaxpr_dict=dict()
    jaxpr_dict["constvars"]=jaxpr.constvars
    jaxpr_dict["invars"]=jaxpr.invars
    jaxpr_dict["outvars"]=jaxpr.outvars
    jaxpr_dict["effects"]=jaxpr.effects
    eqns_list=list()
    for eqn in jaxpr.eqns:
        if(eqn.primitive.name=="custom_jvp_call"):
            eqn_list=SerializeCallJaxpr(eqn)
        else:
            eqn_list=[eqn.invars,eqn.outvars,eqn.primitive,eqn.params,eqn.effects]
        # source_info=source_info_util.new_source_info()
        # source_info=[]
        eqns_list.append(eqn_list)
        # eqn.__setattr__("source_info",None)
    jaxpr_dict["eqns"]=eqns_list
    return jaxpr_dict

def DeserializeJaxpr(jaxpr_dict:Dict) -> Jaxpr:
    eqns=list()
    for eqn in jaxpr_dict["eqns"]:
        if(eqn[2].name=="custom_jvp_call"):
            eqns.append(DeserializeCallJaxpr(eqn))
        else:
            eqns.append(JaxprEqn(*eqn,None))
    return Jaxpr(jaxpr_dict["constvars"],jaxpr_dict["invars"],jaxpr_dict["outvars"],eqns,jaxpr_dict["effects"])

def SerializeCallJaxpr(jaxpr_eqn:JaxprEqn)->List:
    params=dict()
    params["jvp_jaxpr_thunk"]=jaxpr_eqn.params["jvp_jaxpr_thunk"]
    params["call_jaxpr"]=SerializeClosedJaxpr(jaxpr_eqn.params["call_jaxpr"])
    return [jaxpr_eqn.invars,jaxpr_eqn.outvars,jaxpr_eqn.primitive,params,jaxpr_eqn.effects]
    
def DeserializeCallJaxpr(eqn_list:list) -> JaxprEqn:
    eqn_list[3]["call_jaxpr"]=DeserializeClosedJaxpr(eqn_list[3]["call_jaxpr"])
    return JaxprEqn(*eqn_list, None)

def SerializeClosedJaxpr(closed_jaxpr: ClosedJaxpr)->Dict: 
    closed_jaxpr_dict=dict()
    closed_jaxpr_dict["consts"]=closed_jaxpr.consts
    jaxpr_dict=SerializeJaxpr(closed_jaxpr.jaxpr)
    closed_jaxpr_dict["jaxpr"]=jaxpr_dict
    return closed_jaxpr_dict

def DeserializeClosedJaxpr(closed_jaxpr_dict: Dict)->ClosedJaxpr: 
    jaxpr=DeserializeJaxpr(closed_jaxpr_dict["jaxpr"])
    closed_jaxpr=ClosedJaxpr(jaxpr,closed_jaxpr_dict["consts"])
    return closed_jaxpr

def SerializeComputeGradJaxpr(closed_jaxpr: ClosedJaxpr,file_path):
    closed_jaxpr_dict=SerializeClosedJaxpr(closed_jaxpr)
    with open(file_path,"wb") as f:
        dill.dump(closed_jaxpr_dict,f)
    # jaxpr_trans=closed_jaxpr.jaxpr.replace(eqns=eqns_trans)
    # closed_jaxpr_trans=closed_jaxpr.replace(jaxpr=jaxpr_trans)
    # return closed_jaxpr_trans

def DeserializationComputeGradJaxpr(file_path) -> ClosedJaxpr:
    with open(file_path,"rb") as f:
        closed_jaxpr_dict=dill.load(f)
    closed_jaxpr=DeserializeClosedJaxpr(closed_jaxpr_dict)
    return closed_jaxpr

if __name__=="__main__":
    file_path="cyg_test/display/compute_grad_jaxpr.pkl"
    compute_grad_jaxpr=DeserializationComputeGradJaxpr(file_path)
    forward_closed_jaxpr,backward_closed_jaxpr=SliceComputeGradJaxprToForwardAndBackward(compute_grad_jaxpr,2)
    graph=JaxprToDot(backward_closed_jaxpr.jaxpr)
    graph.write("cyg_test/backward_closed_jaxpr.png",format="png")