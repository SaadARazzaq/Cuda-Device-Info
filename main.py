import torch

def check_cuda():
    """
    Checks if CUDA is available in the Python environment and prints information about CUDA devices.
    """
    if torch.cuda.is_available():
        print("CUDA is available.")
        print("Number of CUDA devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("CUDA device", i, ":", torch.cuda.get_device_name(i))
            print("CUDA device", i, "compute capability:", torch.cuda.get_device_capability(i))
            print("CUDA device", i, "memory:", torch.cuda.get_device_properties(i).total_memory / (1024 * 1024), "MB")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda()
