#pragma once

#include "spMat.h"
#include <Eigen/Sparse>

class spMat_check :public spMat
{
public:
	Eigen::SparseMatrix<vType> A;
	spMat_check(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols);
	virtual void assign(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols) override;
	virtual std::vector<vType> MatMul(const std::vector<vType>& vec) override;
};