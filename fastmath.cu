#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fastmath.cuh"
#include <cmath>

Vector2::Vector2(float x, float y)
{
	mm.m128_f32[0] = x;
	mm.m128_f32[1] = y;
}

Vector2::Vector2()
{

}

Vector2 Vector2::operator+(const Vector2& Other)
{
	Vector2 out;
	out.mm = _mm_add_ps(Other.mm, mm);
	return out;
}

Vector2 Vector2::operator-(const Vector2& Other)
{
	Vector2 out;
	out.mm = _mm_sub_ps(Other.mm, mm);
	return out;
}

Vector2 Vector2::operator*(const Vector2& Other)
{
	Vector2 out;
	out.mm = _mm_mul_ps(Other.mm, mm);
	return out;
}

Vector2 Vector2::operator/(const Vector2& Other)
{
	Vector2 out;
	out.mm = _mm_div_ps(Other.mm, mm);
	return out;
}

inline float Vector2::get(unsigned char Component)
{
	return mm.m128_f32[Component];
}

Vector3::Vector3(float x, float y, float z)
{
	mm.m128_f32[0] = x;
	mm.m128_f32[1] = y;
	mm.m128_f32[2] = z;
}

Vector3::Vector3()
{

}

Vector3 Vector3::operator+(const Vector3& Other)
{
	Vector3 out;
	out.mm = _mm_add_ps(Other.mm, mm);
	return out;
}

Vector3 Vector3::operator-(const Vector3& Other)
{
	Vector3 out;
	out.mm = _mm_sub_ps(Other.mm, mm);
	return out;
}

Vector3 Vector3::operator*(const Vector3& Other)
{
	Vector3 out;
	out.mm = _mm_mul_ps(Other.mm, mm);
	return out;
}

Vector3 Vector3::operator/(const Vector3& Other)
{
	Vector3 out;
	out.mm = _mm_div_ps(Other.mm, mm);
	return out;
}

inline float Vector3::get(unsigned char Component)
{
	return mm.m128_f32[Component];
}

Vector4::Vector4(float x, float y, float z, float w)
{
	mm.m128_f32[0] = x;
	mm.m128_f32[1] = y;
	mm.m128_f32[2] = z;
	mm.m128_f32[3] = w;
}

Vector4::Vector4()
{

}

Vector4 Vector4::operator+(const Vector4& Other)
{
	Vector4 out;
	out.mm = _mm_add_ps(Other.mm, mm);
	return out;
}

Vector4 Vector4::operator-(const Vector4& Other)
{
	Vector4 out;
	out.mm = _mm_sub_ps(Other.mm, mm);
	return out;
}

Vector4 Vector4::operator*(const Vector4& Other)
{
	Vector4 out;
	out.mm = _mm_mul_ps(Other.mm, mm);
	return out;
}

Vector4 Vector4::operator/(const Vector4& Other)
{
	Vector4 out;
	out.mm = _mm_div_ps(Other.mm, mm);
	return out;
}

inline float Vector4::get(unsigned char Component)
{
	return mm.m128_f32[Component];
}

float dot(const Vector2& x, const Vector2& y)
{
	__m128 added = _mm_add_ps(x.mm, y.mm);
	return added.m128_f32[0] + added.m128_f32[1];
}

float dot(const Vector3& x, const Vector3& y)
{
	__m128 added = _mm_add_ps(x.mm, y.mm);
	return added.m128_f32[0] + added.m128_f32[1] + added.m128_f32[2];
}

float dot(const Vector4& x, const Vector4& y)
{
	__m128 added = _mm_add_ps(x.mm, y.mm);
	return added.m128_f32[0] + added.m128_f32[1] + added.m128_f32[2] + added.m128_f32[3];
}

Vector3 cross(const Vector3& x, const Vector3& y)
{
	Vector3 result;
	result.mm.m128_f32[0] = x.mm.m128_f32[1] * y.mm.m128_f32[2] - x.mm.m128_f32[2] * y.mm.m128_f32[1];
	result.mm.m128_f32[1] = x.mm.m128_f32[2] * y.mm.m128_f32[0] - x.mm.m128_f32[0] * y.mm.m128_f32[2];
	result.mm.m128_f32[2] = x.mm.m128_f32[0] * y.mm.m128_f32[1] - x.mm.m128_f32[1] * y.mm.m128_f32[0];
	return result;
}

float length(const Vector2& x)
{
	__m128 powof2 = _mm_mul_ps(x.mm, x.mm);
	return sqrtf(powof2.m128_f32[0] + powof2.m128_f32[1]);
}
float length(const Vector3& x)
{
	__m128 powof2 = _mm_mul_ps(x.mm, x.mm);
	return sqrtf(powof2.m128_f32[0] + powof2.m128_f32[1] + powof2.m128_f32[2]);
}
float length(const Vector4& x)
{
	__m128 powof2 = _mm_mul_ps(x.mm, x.mm);
	return sqrtf(powof2.m128_f32[0] + powof2.m128_f32[1] + powof2.m128_f32[2] + powof2.m128_f32[3]);
}

Vector2 normalize(const Vector2& x)
{
	Vector2 result = x;
	float xlen = 1.0f / length(x);
	result.mm.m128_f32[0] *= xlen;
	result.mm.m128_f32[1] *= xlen;
	return result;
}

Vector3 normalize(const Vector3& x)
{
	Vector3 result = x;
	float xlen = 1.0f / length(x);
	result.mm.m128_f32[0] *= xlen;
	result.mm.m128_f32[1] *= xlen;
	result.mm.m128_f32[2] *= xlen;
	return result;
}

Vector4 normalize(const Vector4& x)
{
	Vector4 result = x;
	float xlen = 1.0f / length(x);
	result.mm.m128_f32[0] *= xlen;
	result.mm.m128_f32[1] *= xlen;
	result.mm.m128_f32[2] *= xlen;
	result.mm.m128_f32[3] *= xlen;
	return result;
}

Matrix::Matrix(size_t rows, size_t columns)
{
	h_data = (float*)malloc(rows * columns * sizeof(float));
	cudaMalloc((void**)&d_data, rows * columns * sizeof(float));
	validated = true;
	this->rows = rows;
	this->columns = columns;
}

float Matrix::get(int rowI, int colI)
{
	if (validated) return h_data[rowI * columns + colI];
	cudaMemcpy(h_data, d_data, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
	validated = true;
	return h_data[rowI * columns + colI];
}

void Matrix::set(int rowI, int colI, float val)
{
	if (!validated) cudaMemcpy(h_data, d_data, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
	h_data[rowI * columns + colI] = val;
	validated = true;
}

void Matrix::randomize(float deviation)
{
	deviation *= 2;
	for (int i = 0; i < rows * columns; i++)
	{
		h_data[i] = (rand() % 100000) / (100000.0f / deviation) - (deviation / 2.0f);
	}
}

Matrix Matrix::operator+(Matrix& Other)
{
	Matrix output(rows, columns);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			output.set(i, j, get(i, j) + Other.get(i, j));
		}
	}
	return output;
}

Matrix Matrix::operator-(Matrix& Other)
{
	Matrix output(rows, columns);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			output.set(i, j, get(i, j) - Other.get(i, j));
		}
	}
	return output;
}

__global__ void d_matmul(float* out, float* a, float* b, int c1, int c2, int c3)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;
	for (int k = 0; k < c1; k++)
	{
		out[i * c3 + j] += a[i * c1 + k] * b[k * c2 + j];
	}
}

Matrix matmul(const Matrix& x, const Matrix& y)
{
	Matrix output(x.rows, y.columns);

	cudaMemset(output.d_data, 0, x.rows * y.columns * sizeof(float));

	if (x.validated) cudaMemcpy(x.d_data, x.h_data, x.rows * x.columns * sizeof(float), cudaMemcpyHostToDevice);
	if (y.validated) cudaMemcpy(y.d_data, y.h_data, y.rows * y.columns * sizeof(float), cudaMemcpyHostToDevice);

	d_matmul<<<x.rows, y.columns>>>(output.d_data, x.d_data, y.d_data, x.columns, y.columns, output.columns);

	output.validated = false;

	return output;
}

Matrix transpose(const Matrix& x)
{
	if (!x.validated) cudaMemcpy(x.h_data, x.d_data, x.rows * x.columns * sizeof(float), cudaMemcpyDeviceToHost);

	Matrix output(x.columns, x.rows);

	for (int i = 0; i < x.rows; i++)
	{
		for (int j = 0; j < x.columns; j++)
		{
			output.set(j, i, x.h_data[i * x.columns + j]);
		}
	}

	return output;
}

__global__ void d_mean0(float* output, float* input, int columns)
{
	int i = blockIdx.x;
	output[i] = 0.0f;
	for (int j = 0; j < columns; j++)
	{
		output[i] += input[i * columns + j];
	}
	output[i] /= (float)columns;
}

__global__ void d_mean1(float* output, float* input, int columns, int rows)
{
	int i = blockIdx.x;
	output[i] = 0.0f;
	for (int j = 0; j < rows; j++)
	{
		output[i] += input[j * columns + i];
	}
	output[i] /= (float)rows;
}

Matrix mean(const Matrix& x, int axis)
{
	if (x.validated) cudaMemcpy(x.d_data, x.h_data, x.rows * x.columns * sizeof(float), cudaMemcpyHostToDevice);

	if (axis == 0)
	{
		Matrix output(x.rows, 1);
		d_mean0<<<x.rows, 1>>>(output.d_data, x.d_data, x.columns);
		output.validated = false;
		return output;
	}
	else
	{
		Matrix output(1, x.columns);
		d_mean1<<<x.columns, 1>>>(output.d_data, x.d_data, x.columns, x.rows);
		output.validated = false;
		return output;
	}
}