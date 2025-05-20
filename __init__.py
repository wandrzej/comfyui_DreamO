# ComfyUI will look for these mappings
# Ensure dreamo.dreamo_nodes is accessible
try:
    from .dreamo.dreamo_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
except ImportError as e:
    print(f"[DreamO Custom Node] Failed to import node mappings: {e}")
    print(f"[DreamO Custom Node] Ensure that the 'dreamo' directory and its contents are structured correctly.")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
