#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>

enum class FractalType { MANDELBROT = 0, BURNING_SHIP = 1, TRICORN = 2, JULIA = 3 };

int ITER = 500;
float POW = 0.5;
float RATIO = 0.5;

struct FractalConfig {
    FractalType type;
    std::string name;
    float centerX, centerY, zoom;
    float jCx, jCy;
};

__device__ void applyColoring(float t, unsigned char& r, unsigned char& g, unsigned char& b) {
    t = fmax(0.0f, fmin(1.0f, t));
    if (t < 0.25) {
        float m = t / 0.25;
        r = 68 + m * 3; g = 1 + m * 43; b = 84 + m * 38;
    }
    else if (t < 0.5) {
        float m = (t - 0.25) / 0.25;
        r = 71 - m * 38; g = 44 + m * 100; b = 122 + m * 19;
    }
    else if (t < 0.75) {
        float m = (t - 0.5) / 0.25;
        r = 33; g = 144 + m * 37; b = 141 - m * 68;
    }
    else {
        float m = (t - 0.75) / 0.25;
        r = 33 + m * 220; g = 181 + m * 50; b = 73 - m * 36;
    }
}

void applyColoringCPU(double t, unsigned char& r, unsigned char& g, unsigned char& b) {
    t = fmax(0.0, fmin(1.0, t));
    if (t < 0.25) {
        double m = t / 0.25;
        r = 68 + m * 3; g = 1 + m * 43; b = 84 + m * 38;
    }
    else if (t < 0.5) {
        double m = (t - 0.25) / 0.25;
        r = 71 - m * 38; g = 44 + m * 100; b = 122 + m * 19;
    }
    else if (t < 0.75) {
        double m = (t - 0.5) / 0.25;
        r = 33; g = 144 + m * 37; b = 141 - m * 68;
    }
    else {
        double m = (t - 0.75) / 0.25;
        r = 33 + m * 220; g = 181 + m * 50; b = 73 - m * 36;
    }
}

void fractalCPU(unsigned char* h_image, int w, int h, FractalConfig config, int maxIter, double power, double ratio) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double aspect = (double)w / h;
            double worldX = config.centerX + (x - w / 2.0) * (4.0 / (w * config.zoom)) * aspect;
            double worldY = config.centerY + (y - h / 2.0) * (4.0 / (h * config.zoom));

            double zx, zy, cx, cy;
            if (config.type == FractalType::JULIA) {
                zx = worldX; zy = worldY; cx = config.jCx; cy = config.jCy;
            }
            else {
                zx = 0.0; zy = 0.0; cx = worldX; cy = worldY;
            }

            int iter = 0;
            while (zx * zx + zy * zy <= 4.0 && iter < maxIter) {
                double tmp = zx * zx - zy * zy + cx;
                if (config.type == FractalType::BURNING_SHIP) {
                    zy = fabs(2.0 * zx * zy) + cy;
                    zx = fabs(tmp);
                }
                else if (config.type == FractalType::TRICORN) {
                    zy = -2.0 * zx * zy + cy;
                    zx = tmp;
                }
                else {
                    zy = 2.0 * zx * zy + cy;
                    zx = tmp;
                }
                iter++;
            }

            unsigned char r, g, b;
            if (iter == maxIter) {
                r = (config.type == FractalType::JULIA) ? 255 : 0;
                g = (config.type == FractalType::JULIA) ? 255 : 10;
                b = (config.type == FractalType::JULIA) ? 50 : 25;
            }
            else {
                double mu = iter + 1.0 - log2(log(sqrt(zx * zx + zy * zy)));
                applyColoringCPU(pow(mu / (maxIter * ratio), power), r, g, b);
            }

            int idx = (y * w + x) * 4;
            h_image[idx] = b; h_image[idx + 1] = g; h_image[idx + 2] = r; h_image[idx + 3] = 255;
        }
    }
}

__global__ void fractalKernel(unsigned char* d_ptr, int w, int h, FractalConfig config, int maxIter, float power, float ratio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float aspect = (float)w / h;
    float worldX = config.centerX + (x - w / 2.0) * (4.0 / (w * config.zoom)) * aspect;
    float worldY = config.centerY + (y - h / 2.0) * (4.0 / (h * config.zoom));

    float zx, zy, cx, cy;
    if (config.type == FractalType::JULIA) {
        zx = worldX; zy = worldY; cx = config.jCx; cy = config.jCy;
    }
    else {
        zx = 0.0; zy = 0.0; cx = worldX; cy = worldY;
    }

    int iter = 0;
    while (zx * zx + zy * zy <= 4.0 && iter < maxIter) {
        float tmp = zx * zx - zy * zy + cx;
        if (config.type == FractalType::BURNING_SHIP) {
            zy = fabs(2.0 * zx * zy) + cy;
            zx = fabs(tmp);
        }
        else if (config.type == FractalType::TRICORN) {
            zy = -2.0 * zx * zy + cy;
            zx = tmp;
        }
        else {
            zy = 2.0 * zx * zy + cy;
            zx = tmp;
        }
        iter++;
    }

    unsigned char r, g, b;
    if (iter == maxIter) {
        r = (config.type == FractalType::JULIA) ? 255 : 0;
        g = (config.type == FractalType::JULIA) ? 255 : 10;
        b = (config.type == FractalType::JULIA) ? 50 : 25;
    }
    else {
        float mu = iter + 1.0 - log2(log(sqrt(zx * zx + zy * zy)));
        applyColoring(pow(mu / (maxIter * ratio), power), r, g, b);
    }

    int idx = (y * w + x) * 4;
    d_ptr[idx] = b; d_ptr[idx + 1] = g; d_ptr[idx + 2] = r; d_ptr[idx + 3] = 255;
}

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t type{ 0x4D42 }; uint32_t size, res, offset{ 54 }, dibSize{ 40 };
    int32_t  w, h; uint16_t planes{ 1 }, bpp{ 32 };
    uint32_t comp{ 0 }, imgSize{ 0 }; int32_t hRes{ 2835 }, vRes{ 2835 };
    uint32_t colors{ 0 }, impColors{ 0 };
};
#pragma pack(pop)

template<typename T>
T ask(std::string prompt, T defaultVal) {
    std::cout << prompt << " [" << defaultVal << "]: ";
    std::string input;
    std::getline(std::cin, input);
    if (input.empty()) return defaultVal;
    if constexpr (std::is_same_v<T, int>) return std::stoi(input);
    if constexpr (std::is_same_v<T, float>) return std::stod(input);
    return defaultVal;
}

void parseJulia(float& cx, float& cy) {
    std::cout << "Cx, Cy [" << cx << " " << cy << "]: ";
    std::string input;
    std::getline(std::cin, input);
    if (input.empty()) return;

    for (char& c : input) if (c == ',' || c == '+') c = ' ';
    std::stringstream ss(input);
    ss >> cx >> cy;
}

void parseDimensions(int& w, int& h) {
    std::cout << "RESOLUTION [3840x2160]: ";
    std::string input;
    std::getline(std::cin, input);
    if (input.empty()) { w = 3840; h = 2160; return; }
    for (char& c : input) if (c == 'x' || c == 'X' || c == ',') c = ' ';
    std::stringstream ss(input);
    if (!(ss >> w >> h)) { w = 3840; h = 2160; }
}

void generateFractal(int w, int h, FractalConfig config, bool runCPU)
{
    size_t img_size = (size_t)w * h * 4 * sizeof(unsigned char);
    unsigned char* h_image = (unsigned char*)malloc(img_size);
    if (!h_image) {
        printf("Memory allocation failed.\n");
        return;
    }
    float cpu_time = 0.0;
    if (runCPU) {
        clock_t start_cpu = clock();
        fractalCPU(h_image, w, h, config, ITER, POW, RATIO);
        clock_t end_cpu = clock();
        cpu_time = (float)(end_cpu - start_cpu) * 1000 / CLOCKS_PER_SEC;
        printf("CPU time: %.2f ms\n", cpu_time);
    }

    size_t size = (size_t)w * h * 4;
    unsigned char* h_ptr = (unsigned char*)malloc(size), * d_ptr;
    cudaMalloc(&d_ptr, size);

    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
    std::cout << "\nRendering at " << w << "x" << h << "...\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fractalKernel << <grid, block >> > (d_ptr, w, h, config, ITER, POW, RATIO);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU time: %.2f ms\n", gpu_time);

    if (runCPU) {
        printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    }

    std::string folder_path = "output";
    std::string fname = folder_path + "/" + "frac_" + config.name + ".bmp";

    std::ofstream f(fname, std::ios::binary);
    BMPHeader bh; bh.size = sizeof(BMPHeader) + (uint32_t)size; bh.w = w; bh.h = -h;
    f.write((char*)&bh, sizeof(bh));
    f.write((char*)h_ptr, size);

    std::cout << "Saved: " << fname << std::endl;
    cudaFree(d_ptr); free(h_ptr);
}

bool readYesNo(const char* prompt, bool defaultValue) {
    char buffer[64];
    printf("%s [%s]: ", prompt, defaultValue ? "y" : "n");

    if (!fgets(buffer, sizeof(buffer), stdin)) {
        return defaultValue;
    }

    char first = buffer[0];
    if (first == '\n' || first == '\r') {
        return defaultValue;
    }

    first = tolower(first);
    return (first == 'y');
}

int main() {
    FractalConfig presets[4] = {
        { FractalType::MANDELBROT,   "mandelbrot",  -0.5,  0.0, 1.7 },
        { FractalType::BURNING_SHIP, "burningship", -0.45, -0.55, 1.7 },
        { FractalType::TRICORN,      "tricorn",      0.0,  0.0, 1.1 },
        { FractalType::JULIA,        "julia",        0.0,  0.0, 1.7 }
    };

    std::cout << "--- GPU Fractal Renderer ---\n";
    bool runCPU = readYesNo("Run CPU version for comparison (y/n)?", false);
    std::cout << "0:Mandelbrot  1:Burning Ship  2:Tricorn  3:Julia\n";
    int choice = ask("TYPE", 0);
    FractalConfig config = presets[(choice >= 0 && choice < 4) ? choice : 0];

    int w, h;
    parseDimensions(w, h);

    if (config.type == FractalType::JULIA) {
        struct JuliaPreset { float cx, cy; };
        JuliaPreset jPresets[] = {
            {-0.7, 0.27015},
            {-0.8, 0.156},
            {0.3602, 0.1003},
            {-0.75, 0.11},
            {0, 0.8},
            {0.37, 0.1},
            {0.355, 0.355},
            {-0.54, 0.54},
            {-0.4, -0.59},
            {0.34, -0.05},
            {0.355534, -0.337292},
            {-0.123, 0.745},
            {-0.75, 0},
            {-0.391, -0.587},
            {0, 1},
            {0.285, 0.01}
        };
        int numJPresets = sizeof(jPresets) / sizeof(JuliaPreset);

        for (int i = 0; i < numJPresets; ++i)
        {
            config.name = presets[(choice >= 0 && choice < 4) ? choice : 0].name;
            config.jCx = jPresets[i].cx;
            config.jCy = jPresets[i].cy;

            std::ostringstream _cx;
            _cx.precision(5);
            _cx << std::fixed << config.jCx;

            std::ostringstream _cy;
            _cy.precision(5);
            _cy << std::fixed << config.jCy;

            config.name += "_" + _cx.str() + "_" + _cy.str();

            generateFractal(w, h, config, (runCPU && i == 0));
        }
    }
    else
    {
        generateFractal(w, h, config, runCPU);
    }
}