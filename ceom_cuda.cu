#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>

#define N 160000 // 初始状态数量
#define STATE_LEN 11
#define ACTION_LEN 10
#define DIM 2 // 每个状态或动作的维度（2维）

// const int N = 10;
const int numStates = N;

// 用于初始化 cuRAND 状态的核函数
__global__ void setup_kernel(curandState *state, unsigned long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numStates)
    {
        curand_init(seed, id, 0, &state[id]);
    }
}

// 用于计算状态和损失的核函数
__global__ void computeStateAndLoss(float *states_init, float *states_trajectory, float *actions, curandState *state)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > N)
    {
        return;
    }
    // printf("tid是：%d\n", tid);
    // 获取对应的初始状态
    // 访问当前线程的初始状态
    float init_state[DIM];
    init_state[0] = states_init[tid * DIM];
    init_state[1] = states_init[tid * DIM + 1];
    // printf("initial_state0: %f\n", init_state[0]);

    if (tid >= numStates)
        return;

    // 初始化参数
    const int rows = 10;          // 样本矩阵的行数
    const int cols = 2;           // 样本矩阵的列数
    const int sample_N = 2000;    // 采样数
    const int elite_N = 100;      // 精英样本数
    const float threshold = 1e-6; // 损失变化阈值
    const int iterations = 100;   // 最大迭代次数

    // 定义样本、损失、精英样本和损失的数组
    float samples[sample_N][rows][cols];      // 采样矩阵
    float losses[sample_N];                   // 损失矩阵
    float elite_samples[elite_N][rows][cols]; // 精英样本
    float elite_losses[elite_N];              // 精英损失值

    // 初始化单条状态轨迹数组
    // float state1_list_temp[11] = {0}; // 状态数组
    // float state2_list_temp[11] = {0}; // 状态数组
    float states[rows + 1][cols];

    // 随机数生成器初始化
    // curandState state;
    // curand_init(1234, 0, 0, &state);

    // 初始化 means 和 stds 矩阵
    float d_means[rows * cols];
    float d_stds[rows * cols];
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            d_means[i * cols + j] = curand_normal(state); // 从高斯分布采样
            d_stds[i * cols + j] = 1.0f;                  // 初始 std 为 1
        }
    }

    float pre_loss = 1.0f; // 上一次损失

    // 开始迭代
    for (int iter = 0; iter < iterations; iter++)
    {
        // 采样生成 2000 个样本
        for (int n = 0; n < sample_N; n++)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    samples[n][i][j] = d_means[i * cols + j] + d_stds[i * cols + j] * curand_normal(state);
                }
            }
        }

        // // 计算损失
        // for (int n = 0; n < sample_N; n++)
        // {
        //     float loss = 0.0f;
        //     for (int i = 0; i < rows; i++)
        //     {
        //         for (int j = 0; j < cols; j++)
        //         {
        //             loss += samples[n][i][j] * samples[n][i][j]; // 示例：平方和作为损失
        //         }
        //     }
        //     losses[n] = loss;
        // }

        // 计算损失
        for (int n = 0; n < sample_N; n++)
        {
            float loss = 0.0f;

            states[0][0] = init_state[0];
            states[0][1] = init_state[1];

            for (int i = 0; i < rows; ++i)
            {
                // printf("i is: %d\n", i);
                // 生成正态分布随机数并进行转换

                // 计算状态转移方程
                states[i + 1][0] = -0.3f * states[i][1] * cosf(states[i][0]) + samples[n][i][0];
                states[i + 1][1] = 1.01f * states[i][1] + 0.2f * sinf(states[i][0] * states[i][0]) + samples[n][i][1];

                // 计算损失函数
                loss += states[i][0] * states[i][0] + states[i][1] * states[i][1] + samples[n][i][0] * samples[n][i][0] + samples[n][i][1] * samples[n][i][1];
                // printf("Loss is: %f \n", loss);
            }
            loss += states[10][0] * states[10][0] + states[10][1] * states[10][1];
            losses[n] = loss;
            // printf("Loss is: %f \n", loss);
            // printf("-----------------------------\n");
        }

        // 选择精英样本（选择损失最小的 100 个样本）
        for (int i = 0; i < elite_N; i++)
        {
            for (int j = i + 1; j < sample_N; j++)
            {
                if (losses[j] < losses[i])
                {
                    // 交换损失
                    float temp_loss = losses[i];
                    losses[i] = losses[j];
                    losses[j] = temp_loss;

                    // 交换样本
                    for (int x = 0; x < rows; x++)
                    {
                        for (int y = 0; y < cols; y++)
                        {
                            float temp_sample = samples[i][x][y];
                            samples[i][x][y] = samples[j][x][y];
                            samples[j][x][y] = temp_sample;
                        }
                    }
                }
            }
        }

        // 保存精英样本
        for (int i = 0; i < elite_N; i++)
        {
            elite_losses[i] = losses[i];
            // printf("elite_losses: %f\n", elite_losses[i]);
            for (int x = 0; x < rows; x++)
            {
                for (int y = 0; y < cols; y++)
                {
                    elite_samples[i][x][y] = samples[i][x][y];
                }
            }
        }

        //     // 更新 means 和 stds
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float sum = 0.0f;
                for (int n = 0; n < elite_N; n++)
                {
                    sum += elite_samples[n][i][j];
                }
                d_means[i * cols + j] = sum / elite_N;

                float variance = 0.0f;
                for (int n = 0; n < elite_N; n++)
                {
                    float diff = elite_samples[n][i][j] - d_means[i * cols + j];
                    variance += diff * diff;
                }
                d_stds[i * cols + j] = sqrtf(variance / elite_N);
            }
        }

        // 检查停止条件：损失变化是否小于阈值
        float current_loss = elite_losses[0];
        // printf("pre_loss is: %f\n", pre_loss);
        // printf("current_loss is: %f\n", current_loss);
        // printf("----------");
        if (fabs(pre_loss - current_loss) < threshold)
        {
            // printf("STOP at iter: %d\n", iter);
            // // 输出动作
            // for (int n = 0; n < elite_N; n++)
            // { // 遍历第1维
            //     printf("Elite num %d\n", n);
            //     for (int i = 0; i < rows; i++)
            //     { // 遍历第2维
            //         for (int j = 0; j < cols; j++)
            //         { // 遍历第3维
            //             printf("%f ", elite_samples[n][i][j]);
            //         }
            //         printf("\n");
            //     }
            //     printf("-----------------------------");
            // }
            // 生成trajectory矩阵
            for (int i = 0; i < rows; ++i)
            {
                // 计算状态转移方程
                states[i + 1][0] = -0.3f * states[i][1] * cosf(states[i][0]) + elite_samples[0][i][0];
                states[i + 1][1] = 1.01f * states[i][1] + 0.2f * sinf(states[i][0] * states[i][0]) + elite_samples[0][i][1];
                // 保存状态到轨迹矩阵
                states_trajectory[tid * STATE_LEN * DIM + i * DIM] = states[i][0];
                states_trajectory[tid * STATE_LEN * DIM + i * DIM + 1] = states[i][1];
                // 保存动作到动作矩阵
                actions[tid * ACTION_LEN * DIM + i * DIM] = elite_samples[0][i][0];
                actions[tid * ACTION_LEN * DIM + i * DIM + 1] = elite_samples[0][i][1];
            }
            states_trajectory[tid * STATE_LEN * DIM + rows * DIM] = states[rows][0];
            states_trajectory[tid * STATE_LEN * DIM + rows * DIM + 1] = states[rows][1];
            // // 输出trajecory矩阵
            // printf("状态轨迹为：\n");
            // for (int i = 0; i < rows; i++)
            // { // 遍历第2维
            //     for (int j = 0; j < cols; j++)
            //     { // 遍历第3维
            //         printf("%f ", states[i][j]);
            //     }
            //     printf("\n");
            // }
            // printf("-----------------------------");

            break;
        }
        pre_loss = current_loss;
    }
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // 1. 分配并初始化主机内存
    size_t states_init_size = N * DIM * sizeof(float);
    size_t states_trajectory_size = N * STATE_LEN * DIM * sizeof(float);
    size_t actions_size = N * ACTION_LEN * DIM * sizeof(float);

    float *h_states_init = new float[N * DIM];
    float *h_states_trajectory = new float[N * STATE_LEN * DIM];
    float *h_actions = new float[N * ACTION_LEN * DIM];

    // 初始化初始状态（用户可自定义）
    int idx = 0;
    for (float state0 = -2.0f; state0 <= 2.0f; state0 += 0.01f)
    {
        for (float state1 = -2.0f; state1 <= 2.0f; state1 += 0.01f)
        {
            if (idx < N) // 避免越界
            {
                h_states_init[idx * DIM] = state0;
                h_states_init[idx * DIM + 1] = state1;
                ++idx;
            }
            else
            {
                break; // 达到 N 个元素后退出
            }
        }
    }

    // 2. 分配设备内存
    float *d_states_init, *d_states_trajectory, *d_actions;
    cudaMalloc(&d_states_init, states_init_size);
    cudaMalloc(&d_states_trajectory, states_trajectory_size);
    cudaMalloc(&d_actions, actions_size);

    // 3. 将初始状态从主机复制到设备
    cudaMemcpy(d_states_init, h_states_init, states_init_size, cudaMemcpyHostToDevice);

    // 4. 定义核函数配置
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // 在设备（GPU）上分配 cuRAND 状态数组
    curandState *d_state;
    checkCudaError(cudaMalloc((void **)&d_state, numStates * sizeof(curandState)), "Allocating d_state");

    // 初始化 cuRAND 状态
    setup_kernel<<<(numStates + 255) / 256, 256>>>(d_state, time(NULL));
    // setup_kernel<<<1, 1>>>(d_state, time(NULL));
    checkCudaError(cudaGetLastError(), "Launching setup_kernel");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after setup_kernel");

    // 5. 启动核函数
    computeStateAndLoss<<<blocks_per_grid, threads_per_block>>>(d_states_init, d_states_trajectory, d_actions, d_state);
    checkCudaError(cudaGetLastError(), "Launching computeStateAndLoss");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after computeStateAndLoss");

    // 6. 将结果从设备复制回主机
    cudaMemcpy(h_states_trajectory, d_states_trajectory, states_trajectory_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_actions, d_actions, actions_size, cudaMemcpyDeviceToHost);

    // 7. 打印部分结果验证
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "State trajectory for sample " << i << ":\n";
        for (int j = 0; j < STATE_LEN; ++j)
        {
            std::cout << "("
                      << h_states_trajectory[i * STATE_LEN * DIM + j * DIM] << ", "
                      << h_states_trajectory[i * STATE_LEN * DIM + j * DIM + 1] << ")\n";
        }
        std::cout << "Actions for sample " << i << ":\n";
        for (int j = 0; j < ACTION_LEN; ++j)
        {
            std::cout << "("
                      << h_actions[i * ACTION_LEN * DIM + j * DIM] << ", "
                      << h_actions[i * ACTION_LEN * DIM + j * DIM + 1] << ")\n";
        }
    }

    // 8. 清理内存
    cudaFree(d_states_init);
    cudaFree(d_states_trajectory);
    cudaFree(d_actions);
    delete[] h_states_init;
    delete[] h_states_trajectory;
    delete[] h_actions;

    return 0;
}
