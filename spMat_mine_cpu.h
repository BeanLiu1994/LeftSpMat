#pragma once
#include "spMat.h"

class spMat_mine :public spMat
{
protected:
	int Rows, Cols;
	std::vector<int> OuterStarts;
	std::vector<int> ColIndices;
	std::vector<vType> Values;
public:
	spMat_mine(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols);
	virtual void assign(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols) override;
	virtual std::vector<vType> MatMul(const std::vector<vType>& vec) override;
};