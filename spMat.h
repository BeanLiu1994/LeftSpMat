#pragma once

#include <vector>
#include <array>

typedef double vType;

class spMat
{
public:
	spMat() {};
	virtual void assign(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols) = 0;
	virtual std::vector<vType> MatMul(const std::vector<vType>& b) = 0;

	virtual ~spMat() {};
};
