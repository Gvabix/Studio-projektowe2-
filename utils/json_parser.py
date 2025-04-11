import json


def parse_graph_json(path):
    with open(path) as f:
        data = json.load(f)

    nodes = []
    edges = []

    def traverse(node, parent=None):
        node_id = node["id"]
        nodes.append(node)

        if parent is not None:
            edges.append((parent["id"], node_id))

        for child in node.get("children", []):
            traverse(child, node)

    traverse(data)
    return nodes, edges
