#include "spMat_check.h"

spMat_check::spMat_check(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols)
{
	assign(data, rows, cols);
}
void spMat_check::assign(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols)
{
	std::vector<Eigen::Triplet<vType>> A_data(data.size());
	for (const auto&m : data)
		A_data.emplace_back(std::get<0>(m), std::get<1>(m), std::get<2>(m));
	A.resize(rows, cols);
	A.setFromTriplets(A_data.begin(), A_data.end());
	A.makeCompressed();
}
std::vector<vType> spMat_check::MatMul(const std::vector<vType>& vec)
{
	assert(A.cols() == vec.size());
	Eigen::Map<const Eigen::Matrix<vType, -1, 1>> b(vec.data(), vec.size());
	Eigen::Matrix<vType, -1, 1> tmp = A * b;
	std::vector<vType> ret(vec.size());
	for (int i = 0; i < vec.size(); ++i)
	{
		ret[i] = tmp(i);
	}
	return ret;
}