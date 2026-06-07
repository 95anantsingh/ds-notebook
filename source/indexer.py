"""
Generate a table of contents for the documentation.
"""

import argparse
import os
import re
import sys

DOCS_ROOT = "source/pages"
INDEX_PATH = "source/index.md"

TOC_START = "<!-- TOC START -->"
TOC_END = "<!-- TOC END -->"

DRY_RUN: bool = False
_changes_needed: bool = False


def _mark_change(msg: str) -> None:
    global _changes_needed
    _changes_needed = True
    print(msg)


def is_md_file(filename: str) -> bool:
    """
    Check if a file is a markdown file (excluding index.md)
    """
    return filename.endswith(".md") and filename.lower() != "index.md"


def get_relative_path(path: str) -> str:
    """
    Get the relative path from the DOCS_ROOT directory
    """
    return os.path.join("pages", os.path.relpath(path, DOCS_ROOT).replace("\\", "/"))


def alphanumeric_sort_key(item: str) -> list:
    """
    Custom sort key for alphanumeric sorting.
    Handles numbers properly (e.g., file2.md comes before file10.md)
    """

    # Split the string into parts of letters and numbers
    parts = re.split(r"(\d+)", item.lower())
    # Convert numeric parts to integers for proper sorting
    result = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part)
    return result


def remove_existing_numbers(filename: str) -> str:
    """
    Remove existing number prefixes from filename
    """
    # Remove patterns like "001 ", "01 ", "1 ", "001-", "01-", "1-" from the beginning
    pattern = r"^\d+[\s-]+"
    return re.sub(pattern, "", filename)


def rename_files_with_numbers(path: str) -> None:
    """
    Rename all .md files in a directory with numbered prefixes
    """
    if not os.path.exists(path):
        return

    entries = os.listdir(path)
    md_files = [f for f in entries if is_md_file(f)]

    if not md_files:
        return

    # Sort files alphanumerically (without existing numbers)
    clean_files = []
    for f in md_files:
        clean_name = remove_existing_numbers(f)
        clean_files.append((f, clean_name))

    # Sort by clean names
    clean_files.sort(key=lambda x: alphanumeric_sort_key(x[1]))

    # Determine number of digits needed
    total_files = len(clean_files)
    digits = len(str(total_files))

    # Rename files with proper numbering
    for i, (original_name, clean_name) in enumerate(clean_files, 1):
        old_path = os.path.join(path, original_name)

        # Create new filename with number prefix
        number_prefix = str(i).zfill(digits)
        kebab_name = clean_name.replace(".md", "").lower().replace(" ", "-") + ".md"
        new_name = f"{number_prefix}-{kebab_name}"
        new_path = os.path.join(path, new_name)

        # Only rename if the name would change
        if original_name != new_name:
            if DRY_RUN:
                _mark_change(f"Would rename: {original_name} -> {new_name}")
            else:
                # Handle case where target file already exists
                if os.path.exists(new_path):
                    temp_path = os.path.join(path, f"temp_{i}_{kebab_name}")
                    os.rename(old_path, temp_path)
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    os.rename(temp_path, new_path)
                else:
                    os.rename(old_path, new_path)

                print(f"📝 Renamed: {original_name} -> {new_name}")


def has_md_files_recursive(path: str) -> bool:
    """Check if directory or its subdirectories contain any .md files"""
    entries = os.listdir(path)

    # Check for direct .md files
    md_files = [f for f in entries if is_md_file(f)]
    if md_files:
        return True

    # Check subdirectories recursively
    dirs = [d for d in entries if os.path.isdir(os.path.join(path, d))]
    for d in dirs:
        sub_path = os.path.join(path, d)
        if has_md_files_recursive(sub_path):
            return True

    return False


def rename_all_files_recursively(path: str) -> None:
    """Recursively rename all .md files in all directories"""
    # Rename files in current directory
    rename_files_with_numbers(path)

    # Process subdirectories
    entries = os.listdir(path)
    dirs = [d for d in entries if os.path.isdir(os.path.join(path, d))]

    for d in dirs:
        sub_path = os.path.join(path, d)
        rename_all_files_recursively(sub_path)


def cleanup_orphaned_indexes(path: str) -> None:
    """Delete index.md files from directories that have no content files"""
    for dirpath, dirnames, filenames in os.walk(path):
        if "index.md" in filenames:
            has_content = any(
                f.endswith(".md") and f.lower() != "index.md"
                for _, _, files in os.walk(dirpath)
                for f in files
            )
            if not has_content:
                orphan = os.path.join(dirpath, "index.md")
                os.remove(orphan)
                if not DRY_RUN:
                    print(f"🗑️  Removed orphaned index: {orphan}")


def generate_toc(path: str, depth: int = 0) -> list:
    """
    Generate a table of contents for the given directory.
    If depth is 0, it will create separate toctree sections for each top-level directory.
    If depth is greater than 0, it will generate a single toctree for the entire directory.
    """
    lines = []
    entries = sorted(os.listdir(path), key=alphanumeric_sort_key)

    dirs = [d for d in entries if os.path.isdir(os.path.join(path, d))]

    # For top-level directories, create separate toctree sections
    if depth == 0:
        for d in dirs:
            sub_path = os.path.join(path, d)

            # Only create section if it has .md files
            if has_md_files_recursive(sub_path):
                heading = d.replace("-", " ").replace("_", " ").title()

                # Create a captioned toctree for each top-level directory
                lines.append("```{toctree}")
                lines.append(":maxdepth: 4")  # Allow deeper nesting
                lines.append(f":caption: {heading}")
                lines.append(":glob:")

                lines.append("")

                # Generate content with subsections
                section_lines = generate_section_with_subsections(sub_path)
                lines.extend(section_lines)

                lines.append("```")
                lines.append("")

    return lines


def generate_section_with_subsections(path: str) -> list:
    """
    Generate a section with subsections for the given directory.
    This will add markdown files directly and subdirectories as subsections.
    """
    lines = []
    entries = sorted(os.listdir(path), key=alphanumeric_sort_key)

    md_files = [f for f in entries if is_md_file(f)]
    dirs = [d for d in entries if os.path.isdir(os.path.join(path, d))]

    # Add direct markdown files first (they should now be numbered)
    for f in md_files:
        rel_path = get_relative_path(os.path.join(path, f))
        lines.append(rel_path)

    # Add subdirectories (these become subsections) - only if they have .md files
    for d in dirs:
        sub_path = os.path.join(path, d)

        # Only add subsection if it contains .md files
        if has_md_files_recursive(sub_path):
            # Create an index file for the subsection if it doesn't exist
            create_subsection_index(sub_path, d)

            # Add the subsection to toctree
            rel_path = get_relative_path(os.path.join(sub_path, "index.md"))
            lines.append(rel_path)

    return lines


def create_subsection_index(path: str, section_name: str) -> None:
    """
    Create or update an index.md file for a subsection.

    Only the block between TOC_START / TOC_END is rewritten on each run.
    Content outside those sentinels (heading, prose, custom sections) is
    preserved across runs, so authors can freely annotate subdir indexes.
    """
    index_path = os.path.join(path, "index.md")

    if not has_md_files_recursive(path):
        return

    # Build the toctree entries
    entries = sorted(os.listdir(path), key=alphanumeric_sort_key)
    md_files = [f for f in entries if is_md_file(f)]
    subdirs = [d for d in entries if os.path.isdir(os.path.join(path, d))]

    toctree_content = []
    for f in md_files:
        toctree_content.append(f.replace(".md", ""))
    for d in subdirs:
        sub_path = os.path.join(path, d)
        if has_md_files_recursive(sub_path):
            create_subsection_index(sub_path, d)
            toctree_content.append(f"{d}/index")

    if toctree_content:
        toc_lines = ["```{toctree}", ":maxdepth: 2", ":glob:", ""] + toctree_content + ["```"]
    else:
        toc_lines = []

    toc_block = "\n".join([TOC_START, ""] + toc_lines + ["", TOC_END])

    if not os.path.exists(index_path):
        heading = section_name.replace("-", " ").replace("_", " ").title()
        new_content = f"# {heading}\n\n{toc_block}\n"
    else:
        with open(index_path, "r", encoding="utf-8") as f:
            existing = f.read()
        pattern = re.compile(f"{TOC_START}.*?{TOC_END}", re.DOTALL)
        if TOC_START in existing and TOC_END in existing:
            new_content = pattern.sub(toc_block, existing)
        else:
            new_content = existing.rstrip() + "\n\n" + toc_block + "\n"

    if DRY_RUN:
        existing_check = ""
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                existing_check = f.read()
        if existing_check != new_content:
            _mark_change(f"Would update: {index_path}")
    else:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"📝 Created/Updated subsection index: {index_path}")


def update_index_file() -> None:
    """
    Update the index.md file with the latest table of contents and file renamings.
    """
    if not os.path.exists(INDEX_PATH):
        print(f"❌ index.md not found at {INDEX_PATH}")
        return

    # First, rename all files with numbers
    print("🔄 Renaming files with numbered prefixes...")
    rename_all_files_recursively(DOCS_ROOT)

    print("🔄 Cleaning up orphaned indexes...")
    cleanup_orphaned_indexes(DOCS_ROOT)

    print("🔄 Generating table of contents...")
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    toc_lines = generate_toc(DOCS_ROOT)
    toc_block = "\n".join([TOC_START, ""] + toc_lines + ["", TOC_END])

    pattern = re.compile(f"{TOC_START}.*?{TOC_END}", re.DOTALL)
    if TOC_START in content and TOC_END in content:
        updated_content = pattern.sub(toc_block, content)
    else:
        updated_content = content.strip() + "\n\n" + toc_block

    if DRY_RUN:
        if updated_content != content:
            _mark_change(f"Would update TOC in {INDEX_PATH}")
    else:
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"✅ Files renamed and TOC updated in {INDEX_PATH}")


# Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate table of contents and rename files.")
    parser.add_argument("--check", action="store_true", help="Report changes without making them; exit 1 if any are needed")
    args = parser.parse_args()

    DRY_RUN = args.check
    update_index_file()

    if DRY_RUN:
        if _changes_needed:
            print("❌ Files need updating. Run: python source/indexer.py")
            sys.exit(1)
        else:
            print("✅ All files correctly named and TOC up to date.")
