import json
import os
import shutil
from pathlib import Path

from src.utils.logger import print_section_separator

DEPENDENCIES_SAVE_PATH = Path('submission/dependencies')
SOURCE_CODE_SAVE_PATH = Path('submission/source-code')
SOURCE_CODE_PATH = Path('./')

# You can specify tm hashes here to exclude them from the source code dataset.
TM_HASH = [
    "48e770cd2b34cdf86467014a4315850c",
]


def verify_config():
    print_section_separator("Verify the config files.")
    # Check if kaggle API is setup and installed
    try:
        import kaggle
    except OSError as e:
        new_message = f"To install the Kaggle API key, go to your kaggle profile -> Settings -> Create New API Token."
        raise OSError(new_message) from e

    # Check if the config file is present and filled in
    configs = Path("./submission/config")
    source_path = configs / "source.json"
    dependencies_path = configs / "dependencies.json"

    # load the config files to json
    source_config = json.load(open(source_path))
    dependencies_config = json.load(open(dependencies_path))

    # Check if dependencies is setup
    if source_config["title"] == ".placeholder":
        print("You have not setup the title of the source.json file. Let's do that now.")
        title = input("  - Enter the title of the dataset: ")
        source_config["title"] = title
    if source_config["id"] == ".placeholder/.placeholder":
        print(
            "You have not setup the id of the source.json file. Let's do that now. The username and id can be found in the URL of the dataset.\nFor example, in the URL https://www.kaggle.com/username/dataset-id, the username is 'username' and the id is 'dataset-id'")
        username = input("  - Enter the username of the owner of the source dataset: ")
        id = input("  - Enter the id of the source dataset: ")
        source_config["id"] = f"{username}/{id}"

    # Check if dependencies is setup
    if dependencies_config["title"] == ".placeholder":
        print("You have not setup the title of the dependencies.json file. Let's do that now.")
        title = input("  - Enter the title of the dataset: ")
        dependencies_config["title"] = title
    if dependencies_config["id"] == ".placeholder/.placeholder":
        print(
            "You have not setup the id of the dependencies.json file. Let's do that now. The username and id can be found in the URL of the dataset.\nFor example, in the URL https://www.kaggle.com/username/dataset-id, the username is 'username' and the id is 'dataset-id'")
        username = input("  - Enter the username of the owner of the dependencies dataset: ")
        id = input("  - Enter the id of the dependencies dataset ")
        dependencies_config["id"] = f"{username}/{id}"

    # Write back the updated source_config and dataset_config
    with open(source_path, "w") as f:
        json.dump(source_config, f)
    with open(dependencies_path, "w") as f:
        json.dump(dependencies_config, f)

    print("Config files have been verified.")


def update_dependencies():
    print_section_separator("Update the dependencies.")

    # Automatically compile requirements.txt from pyproject.toml
    print('Compiling requirements.txt from pyproject.toml...')
    compile_cmd = 'uv pip compile pyproject.toml -o requirements.txt'
    result = os.system(compile_cmd)
    if result != 0:
        raise RuntimeError('Failed to compile requirements.txt from pyproject.toml')
    print('✓ requirements.txt compiled successfully')

    # Load excluded packages configuration
    excluded_config_path = Path("./submission/config/excluded_packages.json")
    if excluded_config_path.exists():
        excluded_config = json.load(open(excluded_config_path))
        excluded_packages = excluded_config.get("excluded_packages", [])
        auto_exclude_kaggle = excluded_config.get("auto_exclude_kaggle_packages", False)
        
        print(f'Manually excluded packages: {", ".join(excluded_packages)}')
        
        # Load Kaggle container packages if auto-exclude is enabled
        if auto_exclude_kaggle:
            kaggle_packages_path = Path("./submission/config/kaggle_container_packages.txt")
            if kaggle_packages_path.exists():
                print('Loading Kaggle container packages for auto-exclusion...')
                with open(kaggle_packages_path, 'r') as f:
                    kaggle_lines = f.readlines()
                
                # Parse package names from "package==version" format
                kaggle_packages = []
                for line in kaggle_lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name before ==, >=, etc.
                        pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('!=')[0].strip()
                        if pkg_name:
                            kaggle_packages.append(pkg_name)
                
                # Merge with manually excluded packages (avoid duplicates)
                kaggle_packages_set = set(kaggle_packages)
                excluded_packages_set = set(excluded_packages)
                all_excluded = list(excluded_packages_set.union(kaggle_packages_set))
                
                print(f'  Found {len(kaggle_packages)} packages in Kaggle container')
                print(f'  Total packages to exclude: {len(all_excluded)} (manual + Kaggle)')
                excluded_packages = all_excluded
            else:
                print('  ⚠️  auto_exclude_kaggle_packages is true, but kaggle_container_packages.txt not found')
                print('  Download it by running "pip freeze > kaggle_container_packages.txt" on Kaggle')
    else:
        excluded_packages = []
        print('No excluded packages configuration found.')

    if os.path.exists(DEPENDENCIES_SAVE_PATH):
        print('Cleaning the dependencies folder')
        for filename in os.listdir(DEPENDENCIES_SAVE_PATH):
            file_path = os.path.join(DEPENDENCIES_SAVE_PATH, filename)
            if filename != 'tmp':
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
    else:
        os.makedirs(DEPENDENCIES_SAVE_PATH)

    print('Copying the requirements.txt file and excluding -e, kaggle, and configured packages')
    with open(SOURCE_CODE_PATH / 'requirements.txt', 'r') as f:
        lines = f.readlines()
    with open(DEPENDENCIES_SAVE_PATH / 'requirements.txt', 'w') as f:
        for line in lines:
            line_stripped = line.strip().lower()
            # Skip -e lines
            if line.startswith('-e') or line.lstrip().startswith('#'):
                continue
            # Skip kaggle package
            if line_stripped.startswith('kaggle'):
                continue
            # Skip excluded packages (check package name before == or any operator)
            skip_line = False
            for excluded_pkg in excluded_packages:
                # Check if line starts with the package name (case insensitive)
                if line_stripped.startswith(excluded_pkg.lower()):
                    # Make sure it's followed by ==, >=, etc or end of line
                    pkg_len = len(excluded_pkg)
                    if len(line_stripped) == pkg_len or line_stripped[pkg_len] in ['=', '>', '<', '!', ' ', '\n']:
                        skip_line = True
                        break
            if not skip_line:
                f.write(line)

    if not os.path.exists(DEPENDENCIES_SAVE_PATH / 'tmp'):
        os.makedirs(DEPENDENCIES_SAVE_PATH / 'tmp')
    
    print('Downloading all dependencies for current platform')
    # Simple approach: download for current OS, then filter out platform-specific wheels
    download_cmd = (
        f'uv run pip download '
        f'-r {DEPENDENCIES_SAVE_PATH / "requirements.txt"} '
        f'-d {DEPENDENCIES_SAVE_PATH / "tmp"}'
    )
    result = os.system(download_cmd)
    if result != 0:
        raise RuntimeError('Failed to download dependencies')
    
    print('Filtering out platform-specific (Windows) wheels...')
    tmp_dir = DEPENDENCIES_SAVE_PATH / 'tmp'
    windows_specific_wheels = []
    windows_specific_packages = []  # Track package names to remove from requirements.txt
    
    if os.path.exists(tmp_dir):
        for filename in os.listdir(tmp_dir):
            if not filename.endswith('.whl'):
                # Keep non-wheel files (source distributions are fine)
                continue
            
            # Check if this is a Windows-specific wheel
            # Windows wheels have platform tags like: win32, win_amd64, win_arm64
            # Linux/universal wheels have: manylinux, musllinux, linux, any
            filename_lower = filename.lower()
            
            # List of Windows-specific platform tags
            windows_tags = ['win32', 'win_amd64', 'win_arm64', 'win_ia64']
            is_windows_wheel = any(f'-{tag}.whl' in filename_lower for tag in windows_tags)
            
            if is_windows_wheel:
                windows_specific_wheels.append(filename)
                # Extract package name from wheel filename
                pkg_name = filename.split('-')[0]
                windows_specific_packages.append(pkg_name)
                wheel_path = tmp_dir / filename
                os.remove(wheel_path)
    
    if windows_specific_wheels:
        print(f'  ⚠️  WARNING: Removed {len(windows_specific_wheels)} Windows-specific wheels:')
        for wheel in windows_specific_wheels:
            # Extract package name from wheel filename
            pkg_name = wheel.split('-')[0]
            print(f'    - {pkg_name} ({wheel})')
        print('  These packages will need to be installed from source on Kaggle, or you may need')
        print('  to manually download Linux-compatible wheels for them.')
    else:
        print('  ✓ All wheels are Linux-compatible or platform-independent')

    print('Removing excluded package wheels from downloaded dependencies')
    additional_removed_packages = []  # Track additional packages removed beyond the initial exclusion list
    if excluded_packages:
        tmp_dir = DEPENDENCIES_SAVE_PATH / 'tmp'
        if os.path.exists(tmp_dir):
            removed_count = 0
            for filename in os.listdir(tmp_dir):
                # Check if the wheel file is for an excluded package
                for excluded_pkg in excluded_packages:
                    # Wheel files are named like: package_name-version-...-.whl
                    # where version starts with a digit
                    # Normalize both to lowercase and replace _ with - for comparison
                    normalized_filename = filename.lower().replace('_', '-')
                    normalized_pkg = excluded_pkg.lower().replace('_', '-')
                    
                    # Check if filename starts with package name followed by '-' and a digit (version)
                    # This prevents 'torch' from matching 'torch-ema'
                    if normalized_filename.startswith(normalized_pkg + '-'):
                        # Verify the character after the package name and hyphen is a digit (version number)
                        char_after_pkg = normalized_filename[len(normalized_pkg) + 1:len(normalized_pkg) + 2]
                        if char_after_pkg and char_after_pkg.isdigit():
                            wheel_path = tmp_dir / filename
                            os.remove(wheel_path)
                            print(f'  Removed: {filename}')
                            removed_count += 1
                            # Track the actual package name from the wheel
                            pkg_name = filename.split('-')[0]
                            additional_removed_packages.append(pkg_name)
                            break
            print(f'Total wheels removed: {removed_count}')
    
    # Now update requirements.txt to remove packages whose wheels were removed
    print('Syncing requirements.txt with available wheels...')
    all_removed_packages = set(windows_specific_packages + additional_removed_packages)
    
    if all_removed_packages:
        # Read current requirements.txt
        with open(DEPENDENCIES_SAVE_PATH / 'requirements.txt', 'r') as f:
            req_lines = f.readlines()
        
        # Filter out removed packages
        updated_req_lines = []
        removed_from_req = []
        for line in req_lines:
            line_stripped = line.strip().lower()
            if not line_stripped or line_stripped.startswith('#'):
                updated_req_lines.append(line)
                continue
            
            # Extract package name from requirement line
            req_pkg_name = line_stripped.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('!=')[0].strip()
            req_pkg_name_normalized = req_pkg_name.replace('_', '-')
            
            # Check if this package had its wheel removed
            should_remove = False
            for removed_pkg in all_removed_packages:
                removed_pkg_normalized = removed_pkg.lower().replace('_', '-')
                if req_pkg_name_normalized == removed_pkg_normalized:
                    should_remove = True
                    removed_from_req.append(req_pkg_name)
                    break
            
            if not should_remove:
                updated_req_lines.append(line)
        
        # Write back the updated requirements.txt
        with open(DEPENDENCIES_SAVE_PATH / 'requirements.txt', 'w') as f:
            f.writelines(updated_req_lines)
        
        if removed_from_req:
            print(f'  Removed {len(removed_from_req)} packages from requirements.txt to match available wheels')
            for pkg in removed_from_req:
                print(f'    - {pkg}')
    else:
        print('  ✓ No packages need to be removed from requirements.txt')

    print('Zipping the downloaded dependencies')
    shutil.make_archive(DEPENDENCIES_SAVE_PATH / 'dependencies', 'zip', DEPENDENCIES_SAVE_PATH / 'tmp')
    shutil.move(DEPENDENCIES_SAVE_PATH / 'dependencies.zip', DEPENDENCIES_SAVE_PATH / 'dependencies.no_unzip')
    shutil.rmtree(DEPENDENCIES_SAVE_PATH / 'tmp')

    print('Copying the dataset-metadata.json file')
    shutil.copy('submission/config/dependencies.json', DEPENDENCIES_SAVE_PATH / 'dataset-metadata.json')

    print('Excluding --find-files in requirements.txt')
    with open(DEPENDENCIES_SAVE_PATH / 'requirements.txt', 'r') as f:
        lines = f.readlines()
    with open(DEPENDENCIES_SAVE_PATH / 'requirements.txt', 'w') as f:
        for line in lines:
            if line.startswith('--find-links'):
                continue
            f.write(line)

    print('Done')

    # Upload the dataset
    result = os.system(f'kaggle datasets version -p {DEPENDENCIES_SAVE_PATH} -m "Update Dependencies"')
    if result != 0:
        raise RuntimeError('Failed to upload dependencies dataset to Kaggle')


def update_source():
    """Update the source code dataset."""
    if os.path.exists(SOURCE_CODE_SAVE_PATH):
        shutil.rmtree(SOURCE_CODE_SAVE_PATH)
    os.mkdir(SOURCE_CODE_SAVE_PATH)

    # Copy Source Code to submission/source_code
    relevant_files = ['src/', 'conf/', 'submit.py']
    if len(TM_HASH) == 0:
        relevant_files.append('tm/')
    else:
        for hash in TM_HASH:
            found_one = False
            tm = os.listdir(SOURCE_CODE_PATH / 'tm')
            for file in tm:
                if file.startswith(hash):
                    found_one = True
                    relevant_files.append('tm/' + file)
            if not found_one:
                print(f'No files found with hash: {hash}')
                exit(1)

    # Exclude __pycache__ from copying
    exluded_files = ['__pycache__']

    # Copy relevant files to tmp
    for file in relevant_files:
        if os.path.isdir(SOURCE_CODE_PATH / file):
            # Copy directory, skip excluded files with shutil
            shutil.copytree(SOURCE_CODE_PATH / file, SOURCE_CODE_SAVE_PATH / "tmp" / file, ignore=shutil.ignore_patterns(*exluded_files))
        else:
            # Copy file and create directories if not exist
            os.makedirs(SOURCE_CODE_SAVE_PATH / "tmp" / os.path.dirname(file), exist_ok=True)
            shutil.copy(SOURCE_CODE_PATH / file, SOURCE_CODE_SAVE_PATH / "tmp" / file)

    # Zip source_code
    shutil.make_archive(SOURCE_CODE_SAVE_PATH / 'source-code', 'zip', SOURCE_CODE_SAVE_PATH / "tmp")
    shutil.rmtree(SOURCE_CODE_SAVE_PATH / "tmp")

    # Copy dataset-metadata.json to submission
    shutil.copy('submission/config/source.json', SOURCE_CODE_SAVE_PATH / 'dataset-metadata.json')
    
    # Copy installation scripts and documentation
    if os.path.exists('submission/install_dependencies.sh'):
        shutil.copy('submission/install_dependencies.sh', SOURCE_CODE_SAVE_PATH / 'install_dependencies.sh')
    if os.path.exists('submission/KAGGLE_SETUP.md'):
        shutil.copy('submission/KAGGLE_SETUP.md', SOURCE_CODE_SAVE_PATH / 'KAGGLE_SETUP.md')

    print('Submission files saved to source_code')

    result = os.system(f'kaggle datasets version -p {SOURCE_CODE_SAVE_PATH} -m "Update Source Code"')
    if result != 0:
        raise RuntimeError('Failed to upload source code dataset to Kaggle')


def manage_datasets():
    # Verify the config
    verify_config()

    # Update dependencies
    update_dep = input("Would you like to update the dependencies? (requirements.txt will be auto-compiled from pyproject.toml) (y/n): ").lower()

    if update_dep == "y":
        update_dependencies()

    # Update the dataset
    update_s = input("Would you like to update the source code? (y/n): ").lower()
    if update_s == "y":
        update_source()


if __name__ == "__main__":
    manage_datasets()
