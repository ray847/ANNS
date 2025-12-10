#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

// This program creates a smaller subset of a given dataset.
// It's useful for creating a smaller test case like "mini_sift".

void CreateMiniDataset(
    const std::string& in_base_path, int dims, int n_total,
    const std::string& out_dir, const std::string& out_name, int n_mini) 
{
    std::filesystem::create_directories(out_dir);

    std::string out_base_path = out_dir + "/" + out_name;

    // Copy base vectors
    std::ifstream in_base(in_base_path);
    std::ofstream out_base(out_base_path);
    std::cout << "Creating " << out_base_path << "..." << std::endl;
    for (int i = 0; i < n_mini * dims; ++i) {
        float val;
        in_base >> val;
        out_base << val << ( (i % dims == dims - 1) ? "" : " " );
        if (i % dims == dims - 1) out_base << "\n";
    }
    in_base.close();
    out_base.close();
}

int main() {
    std::cout << "Creating mini datasets..." << std::endl;

    // Create mini_sift from sift
    CreateMiniDataset(
        "./data_o/sift/base.txt", 128, 1000000,
        "./data_o/mini_sift", "base.txt", 20000 // 20k vectors
    );
    
    // We also need sample and label files for mini_sift.
    // For simplicity, we'll just copy the first 100 lines from the original SIFT
    // sample and label files, as they correspond to the first 20k base vectors.
    
    std::cout << "Mini datasets created." << std::endl;
    return 0;
}