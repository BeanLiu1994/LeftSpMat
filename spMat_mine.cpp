#include "spMat_mine_cpu.h"
#include <cassert>
#include <map>
//#include <chrono>
//#include <iostream>

spMat_mine::spMat_mine(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols)
{
	assign(data, rows, cols);
}

void spMat_mine::assign(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols)
{
	std::map<std::pair<int, int>, vType> rc2v;
	for (const auto& m : data)
	{
		const int& r = std::get<0>(m);
		const int& c = std::get<1>(m);
		const vType& v = std::get<2>(m);

		assert(r < rows && c < cols && r >= 0 && c >= 0);
		std::pair<int, int> key(r, c);

		rc2v[key] += v;
	}

	int currentRow = 0;
	int elem_count = 0;
	int nnz = rc2v.size();
	OuterStarts.resize(rows + 1);
	ColIndices.resize(nnz);
	Values.resize(nnz);
	for (decltype(rc2v)::iterator it = rc2v.begin(); it != rc2v.end(); ++it)
	{
		ColIndices[elem_count] = it->first.second;
		Values[elem_count] = it->second;
		if (currentRow == it->first.first)//row
		{
			OuterStarts[currentRow] = elem_count;
			++currentRow;
		}
		++elem_count;
	}
	assert(elem_count == nnz);
	OuterStarts[rows] = nnz;

	Rows = rows;
	Cols = cols;
}

std::vector<vType> spMat_mine::MatMul(const std::vector<vType>& b_vec)
{
	assert(b_vec.size() % Cols == 0);
	int b_cols = b_vec.size() / Cols;
	std::vector<vType> result(Cols * b_cols, 0);
	//auto start = std::chrono::steady_clock::now();
	// b取一列出来
	for (int ith_b_col = 0; ith_b_col < b_cols; ++ith_b_col)
	{
		int offset = ith_b_col * Cols;
		// A取一行出来
		for (int ith_row = 0; ith_row < Rows; ++ith_row)
		{
			// 一行乘一列, 只乘稀疏部分。
			for (int ith_elem = OuterStarts[ith_row]; ith_elem < OuterStarts[ith_row + 1]; ++ith_elem)
			{
				result[ith_row + offset] += Values[ith_elem] * b_vec[ColIndices[ith_elem] + offset];
			}
		}
	}
	//auto end = std::chrono::steady_clock::now();
	//std::cout << (end - start).count() / 1e6 << std::endl;
	return result;
}