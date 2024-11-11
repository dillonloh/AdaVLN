import math
import os
import bpy

# Set rendering engine to Cycles
bpy.context.scene.render.engine = 'CYCLES'

# Define paths
dataset_dir = "example_dataset"
input_glb_dir = os.path.join(dataset_dir, "original") 
output_usd_dir = os.path.join(dataset_dir, "merged")

def merge_mesh(input_glb_path, output_usd_path):
    # Clear all objects in the scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import GLB file
    bpy.ops.import_scene.gltf(filepath=input_glb_path)

    # Join all imported objects into a single mesh and disable shadow casting
    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':  # Check if object is a mesh
            obj.select_set(True)  # Select the mesh object
            # Disable shadow visibility
            if hasattr(obj, "cycles_visibility"):
                obj.cycles_visibility.shadow = False
        else:
            obj.select_set(False)  # Deselect non-mesh objects

    # Join all selected mesh objects into one
    bpy.ops.object.join()

    # Rename the merged object to "Building"
    merged_object = bpy.context.object
    merged_object.name = "Building"

    # Directly set rotation in radians
    merged_object.rotation_mode = 'XYZ'
    merged_object.rotation_euler = (math.radians(-90), 0, 0)

    # Apply the rotation to make it permanent
    bpy.ops.object.transform_apply(rotation=True)

    # Create a new collection to represent the /World prim
    world_collection = bpy.data.collections.new(name="World")
    bpy.context.scene.collection.children.link(world_collection)

    # Move the Building object to the World collection
    if merged_object.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(merged_object)
    world_collection.objects.link(merged_object)

    # Export as USD without specifying root_prim
    bpy.ops.wm.usd_export(filepath=output_usd_path, export_textures=True, root_prim_path='/World')

    print(f"USD file exported with Building under /World at {output_usd_path}")

if __name__ == "__main__":
    for file in os.listdir(input_glb_dir):
        if file.endswith(".glb"):
            print(f"Merging {file}...")
            input_glb_path = os.path.join(input_glb_dir, file)
            output_usd_path = os.path.join(output_usd_dir, file.replace(".glb", ".usd"))
            merge_mesh(input_glb_path, output_usd_path)

    print("All GLB files merged and exported as USD files.")
