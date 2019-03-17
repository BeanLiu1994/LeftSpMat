#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "spMat.h"
#include "spMat_check.h"
#include "spMat_mine_cpu.h"
#include "spMat_mine_gpu.h"

std::tuple<std::vector<std::tuple<int, int, vType>>, int, int>
LoadspA(const std::string& spA_path)
{
	std::ifstream file(spA_path);
	if (!file)
	{
		throw std::runtime_error("can't load file.");
	}
	std::vector<std::tuple<int, int, vType>> ret;
	int rows, cols;
	file >> rows >> cols;

	int r, c;
	vType v;
	while (file >> r >> c >> v)
	{
		ret.emplace_back(r, c, v);
	}
	return std::make_tuple(std::move(ret), rows, cols);
}

std::vector<vType>
Loadb(const std::string& b_path)
{
	std::ifstream file(b_path);
	if (!file)
	{
		throw std::runtime_error("can't load file.");
	}
	std::vector<vType> ret;

	vType v;
	while (file >> v)
	{
		ret.emplace_back(v);
	}
	return ret;
}


vType diff(const std::vector<vType>& v1, const std::vector<vType>& v2)
{
	assert(v1.size() == v2.size());
	vType diff_count = 0;
	for (int i = 0; i < v1.size(); ++i)
	{
		diff_count += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	return diff_count;
}

void saveV(const std::vector<vType>& vec, const std::string& v_path)
{
	std::ofstream file(v_path);
	if (!file)
	{
		throw std::runtime_error("can't load file.");
	}
	std::vector<vType> ret;

	for (const auto& m : vec)
	{
		file << m << std::endl;
	}
}


int main()
{
	// 尝试乘 nx2 的矩阵, 矩阵按整列堆叠在一起
	auto A_data = LoadspA("spA.txt");
	auto b_data = Loadb("b.txt");
	auto b2_data = b_data;
	std::for_each(b_data.begin(), b_data.end(), [](vType& v) {v += rand() * 1000; });
	b2_data.insert(b2_data.end(), b_data.begin(), b_data.end());

	spMat_check checker(std::get<0>(A_data), std::get<1>(A_data), std::get<2>(A_data));
	auto checker_result = checker.MatMul(b2_data);

	spMat_mine mine(std::get<0>(A_data), std::get<1>(A_data), std::get<2>(A_data));
	auto my_result = mine.MatMul(b2_data);
	vType my_diff_result = diff(checker_result, my_result);
	std::cout << my_diff_result << std::endl;

	spMat_mine_gpu mine_gpu(std::get<0>(A_data), std::get<1>(A_data), std::get<2>(A_data));
	auto my_gpu_result = mine_gpu.MatMul(b2_data);
	vType my_gpu_diff_result = diff(checker_result, my_gpu_result);
	std::cout << my_gpu_diff_result << std::endl;

	//saveV(my_result, "cpu.txt");
	//saveV(my_gpu_result, "gpu.txt");

	return 0;
}