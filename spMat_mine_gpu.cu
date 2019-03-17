#include "spMat_mine_gpu.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>

const static int blockSize = 256;

__global__ void spmat_mul_vec(vType* d_result,
	const int* d_OuterStarts, const int* d_ColIndices, const vType* d_Values, const vType* d_vec,
	const int rows, const int b_rows)
{
	unsigned int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int offset = threadIdx.y * b_rows;
	if (row_idx >= rows)
	{
		return;
	}

	vType tmp_val = 0;
	for (int ith_elem = d_OuterStarts[row_idx]; ith_elem < d_OuterStarts[row_idx + 1]; ++ith_elem)
	{
		tmp_val += d_Values[ith_elem] * d_vec[d_ColIndices[ith_elem] + offset];
	}
	d_result[row_idx + offset] = tmp_val;
}

// 建议做个智能指针之类的存，那样的话出异常大概可能也许会比较安全
// 这里就直接c风格分空间了

void CudaCheck()
{
	if (cudaGetLastError() != cudaSuccess)
	{
		throw std::runtime_error("Cuda Failed");
	}
}

spMat_mine_gpu::spMat_mine_gpu(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols)
	:spMat_mine(data, rows, cols)
{
	assign_space_and_cpy_to_gpu();
}

void spMat_mine_gpu::assign(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols)
{
	spMat_mine::assign(data, rows, cols);
	assign_space_and_cpy_to_gpu();
}

std::vector<vType> spMat_mine_gpu::MatMul(const std::vector<vType>& vec)
{
	assert(vec.size() % Cols == 0);
	int b_cols = vec.size() / Cols;
	vType* d_result, *d_vec;

	cudaMalloc(&d_vec, vec.size() * sizeof(vType));
	cudaMemcpy(d_vec, vec.data(), vec.size() * sizeof(vType), cudaMemcpyHostToDevice);

	cudaMalloc(&d_result, vec.size() * sizeof(vType));

	// 共有 Rows*b_cols 个任务
	int numThreads = std::min(blockSize, Rows);
	int numBlocks = (Rows % numThreads != 0) ? (Rows / numThreads + 1) : (Rows / numThreads);

	dim3 grid(numBlocks, 1, 1), block(numThreads, b_cols, 1);
	spmat_mul_vec << <grid, block >> > (d_result, _d_OuterStarts, _d_ColIndices, _d_Values, d_vec, Rows, Cols);

	std::vector<vType> ret(vec.size());
	cudaMemcpy(ret.data(), d_result, vec.size() * sizeof(vType), cudaMemcpyDeviceToHost);

	cudaFree(d_vec);
	cudaFree(d_result);
	CudaCheck();
	return ret;
}

spMat_mine_gpu::~spMat_mine_gpu()
{
	free_space_gpu();
}

void spMat_mine_gpu::assign_space_and_cpy_to_gpu()
{
	if (_d_ColIndices || _d_OuterStarts || _d_Values)
	{
		free_space_gpu();
	}
	cudaMalloc(&_d_OuterStarts, OuterStarts.size() * sizeof(decltype(OuterStarts)::value_type));
	cudaMalloc(&_d_ColIndices, ColIndices.size() * sizeof(decltype(ColIndices)::value_type));
	cudaMalloc(&_d_Values, Values.size() * sizeof(decltype(Values)::value_type));

	cudaMemcpy(_d_OuterStarts, OuterStarts.data(), OuterStarts.size() * sizeof(decltype(OuterStarts)::value_type), cudaMemcpyHostToDevice);
	cudaMemcpy(_d_ColIndices, ColIndices.data(), ColIndices.size() * sizeof(decltype(ColIndices)::value_type), cudaMemcpyHostToDevice);
	cudaMemcpy(_d_Values, Values.data(), Values.size() * sizeof(decltype(Values)::value_type), cudaMemcpyHostToDevice);

	CudaCheck();
}
void spMat_mine_gpu::free_space_gpu()
{
	if (_d_ColIndices)
	{
		cudaFree(_d_ColIndices);
		_d_ColIndices = nullptr;
	}
	if (_d_OuterStarts)
	{
		cudaFree(_d_OuterStarts);
		_d_OuterStarts = nullptr;
	}
	if (_d_Values)
	{
		cudaFree(_d_Values);
		_d_Values = nullptr;
	}
	CudaCheck();
}