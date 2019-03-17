#pragma once

#include "spMat.h"
#include "spMat_mine_cpu.h"

// ����cuda��ص�ʵ���ڶ�Ӧ��.cu�ڲ���
class spMat_mine_gpu : public spMat_mine
{
protected:
	int* _d_OuterStarts = nullptr;
	int* _d_ColIndices = nullptr;
	vType* _d_Values = nullptr;

	void assign_space_and_cpy_to_gpu();
	void free_space_gpu();
public:
	spMat_mine_gpu(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols);
	virtual void assign(const std::vector<std::tuple<int, int, vType>>& data, int rows, int cols) override;
	virtual std::vector<vType> MatMul(const std::vector<vType>& vec) override;
	virtual ~spMat_mine_gpu() override;
};