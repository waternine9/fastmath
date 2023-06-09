#ifndef FAST_MATH_H
#define FAST_MATH_H

#include <xmmintrin.h>

class Vector2
{
public:
	__m128 mm;

	__declspec(dllexport) Vector2();
	__declspec(dllexport) Vector2(float x, float y);

	__declspec(dllexport) Vector2 operator+(const Vector2& Other);
	__declspec(dllexport) Vector2 operator-(const Vector2& Other);
	__declspec(dllexport) Vector2 operator*(const Vector2& Other);
	__declspec(dllexport) Vector2 operator/(const Vector2& Other);

	__declspec(dllexport) inline float get(unsigned char Component);
};

class Vector3
{
public:
	__m128 mm;

	__declspec(dllexport) Vector3();
	__declspec(dllexport) Vector3(float x, float y, float z);

	__declspec(dllexport) Vector3 operator+(const Vector3& Other);
	__declspec(dllexport) Vector3 operator-(const Vector3& Other);
	__declspec(dllexport) Vector3 operator*(const Vector3& Other);
	__declspec(dllexport) Vector3 operator/(const Vector3& Other);

	__declspec(dllexport) inline float get(unsigned char Component);
};

class Vector4
{
public:
	__m128 mm;

	__declspec(dllexport) Vector4();
	__declspec(dllexport) Vector4(float x, float y, float z, float w);

	__declspec(dllexport) Vector4 operator+(const Vector4& Other);
	__declspec(dllexport) Vector4 operator-(const Vector4& Other);
	__declspec(dllexport) Vector4 operator*(const Vector4& Other);
	__declspec(dllexport) Vector4 operator/(const Vector4& Other);

	__declspec(dllexport) inline float get(unsigned char Component);
};

__declspec(dllexport) float dot(const Vector2& x, const Vector2& y);
__declspec(dllexport) float dot(const Vector3& x, const Vector3& y);
__declspec(dllexport) float dot(const Vector4& x, const Vector4& y);

__declspec(dllexport) Vector3 cross(const Vector3& x, const Vector3& y);

__declspec(dllexport) float length(const Vector2& x);
__declspec(dllexport) float length(const Vector3& x);
__declspec(dllexport) float length(const Vector4& x);

__declspec(dllexport) Vector2 normalize(const Vector2& x);
__declspec(dllexport) Vector3 normalize(const Vector3& x);
__declspec(dllexport) Vector4 normalize(const Vector4& x);

class Matrix
{
public:
	float* d_data;
	float* h_data;
	size_t rows;
	size_t columns;
	bool validated;
	
	__declspec(dllexport) Matrix(size_t rows, size_t columns);

	__declspec(dllexport) float get(int rowI, int colI);
	__declspec(dllexport) void set(int rowI, int colI, float val);

	__declspec(dllexport) void randomize(float deviation);

	__declspec(dllexport) Matrix operator+(Matrix& Other);
	__declspec(dllexport) Matrix operator-(Matrix& Other);
};

__declspec(dllexport) Matrix matmul(const Matrix& x, const Matrix& y);
__declspec(dllexport) Matrix transpose(const Matrix& x);

/*
* axis 0: average the columns together
* axis 1: average the rows together
*/
__declspec(dllexport) Matrix mean(const Matrix& x, int axis);

#endif // FAST_MATH_H