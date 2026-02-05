import os
import glob
import subprocess
import shutil
import pypandoc

def convert_notebooks(root_dir):
    print(f"Searching for notebooks in {root_dir}...")
    notebooks = glob.glob(os.path.join(root_dir, "**/*.ipynb"), recursive=True)

    print(f"Found {len(notebooks)} notebooks. Starting conversion...")

    for nb_path in notebooks:
        dir_name = os.path.dirname(nb_path)
        base_name = os.path.basename(nb_path)
        name_no_ext = os.path.splitext(base_name)[0]
        md_path = os.path.join(dir_name, name_no_ext + ".md")

        print(f"Converting NB: {nb_path} -> {md_path}")
        try:
            cmd = ["jupyter", "nbconvert", "--to", "markdown", nb_path]
            subprocess.run(cmd, check=True, cwd=dir_name, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {nb_path}: {e}")

def convert_rst(root_dir):
    print("Checking for pandoc...")
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        print("Pandoc not found. Downloading...")
        pypandoc.download_pandoc()

    print(f"Searching for RST files in {root_dir}...")
    rst_files = glob.glob(os.path.join(root_dir, "**/*.rst"), recursive=True)

    print(f"Found {len(rst_files)} RST files. Starting conversion...")

    for rst_path in rst_files:
        dir_name = os.path.dirname(rst_path)
        base_name = os.path.basename(rst_path)
        name_no_ext = os.path.splitext(base_name)[0]
        md_path = os.path.join(dir_name, name_no_ext + ".md")

        print(f"Converting RST: {rst_path} -> {md_path}")
        try:
            pypandoc.convert_file(rst_path, 'commonmark', format='rst', outputfile=md_path)
        except Exception as e:
            print(f"Error converting {rst_path}: {e}")

def create_index(root_dir):
    # 簡単なインデックスファイルを作成
    index_path = os.path.join(root_dir, "README_DOCS.md")
    with open(index_path, "w") as f:
        f.write("# Sionna v1.2.1 Documentation Index\n\n")

        md_files = glob.glob(os.path.join(root_dir, "**/*.md"), recursive=True)
        md_files.sort()

        for md in md_files:
            if md == index_path: continue
            rel_path = os.path.relpath(md, root_dir)
            f.write(f"- [{rel_path}]({rel_path})\n")
    print(f"Created index at {index_path}")

if __name__ == "__main__":
    docs_dir = os.path.join(os.path.dirname(__file__), "../01_docs/sionna-v1.2.1")
    docs_dir = os.path.abspath(docs_dir)

    if not os.path.exists(docs_dir):
        print(f"Directory not found: {docs_dir}")
    else:
        convert_notebooks(docs_dir)
        convert_rst(docs_dir)
        create_index(docs_dir)
