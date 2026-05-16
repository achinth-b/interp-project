import subprocess
import os

NFS_NAME = "vpd-smolvlm-nfs"

def download_file(remote_path: str, local_path: str):
    print(f"Downloading {remote_path} from Modal nfs '{NFS_NAME}'...")
    try:
        # Use --force to overwrite local files
        subprocess.run(["modal", "nfs", "get", "--force", NFS_NAME, remote_path, local_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {remote_path}: {e}")

def download_heatmaps():
    print(f"Checking for heatmaps in Modal nfs '{NFS_NAME}'...")
    # List files to find heatmaps
    try:
        # Note: modal nfs ls output can vary, but we look for filenames
        result = subprocess.run(["modal", "nfs", "ls", NFS_NAME], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            # Modal ls output usually contains the filename as the last element
            parts = line.split()
            if not parts: continue
            filename = parts[-1]
            if "heatmap_atom_" in filename:
                download_file(filename, filename)
    except Exception as e:
        print(f"Error listing heatmaps: {e}")

if __name__ == "__main__":
    # Ensure we are in the project root
    download_file("vpd_checkpoint.pt", "vpd_checkpoint.pt")
    download_heatmaps()
    print("\n✓ Finished downloading all results!")
