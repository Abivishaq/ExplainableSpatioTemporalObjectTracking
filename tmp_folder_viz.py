import os

def print_tree(root_dir, indent="", max_files=10):
    try:
        items = sorted(os.listdir(root_dir))
    except PermissionError:
        print(indent + "|-- [Permission Denied]")
        return

    dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
    files = [item for item in items if not os.path.isdir(os.path.join(root_dir, item))]

    for d in dirs:
        print(indent + "|-- " + d + "/")
        print_tree(os.path.join(root_dir, d), indent + "    ", max_files)

    for i, f in enumerate(files):
        if i < max_files:
            print(indent + "|-- " + f)
        elif i == max_files:
            print(indent + "|-- ...")
            break


your_dataset_path = "data/HOMER"
print_tree(your_dataset_path)
