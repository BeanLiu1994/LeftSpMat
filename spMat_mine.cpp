#include "spMat_mine_cpu.h"
#include <cassert>
#include <map>

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

std::vector<vType> spMat_mine::MatMul(const std::vector<vType>& vec)
{
	assert(vec.size() == Cols);
	std::vector<vType> result(Cols, 0);
	for (int ith_row = 0; ith_row < Rows; ++ith_row)
	{
		// 一行乘一列, 只乘稀疏部分。
		for (int ith_elem = OuterStarts[ith_row]; ith_elem < OuterStarts[ith_row + 1]; ++ith_elem)
		{
			result[ith_row] += Values[ith_elem] * vec[ColIndices[ith_elem]];
		}
	}
	return result;
}