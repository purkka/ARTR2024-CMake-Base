#pragma once

#include <auto_vk_toolkit.hpp>

class simple_geometry
{
public:
	simple_geometry(avk::queue* aQueue) : mQueue{ aQueue } {}

	enum class flags {
		none      = 0x00,
		normals   = 0x01,
		texCoords = 0x02,
		tangents  = 0x04,
		all = normals | texCoords | tangents
	};

	// TODO: support normals, uv ?

	// spheres, cubes are created with radius (halfsize) == 1.0
	// cones are created with height and radius == 1.0, apex at the origin, base at y = +1.0

	void create_cone(int subdivision = 20, bool closedBase = true, glm::mat4 applyTransform = glm::mat4(1));
	void create_sphere(int subdivisionVertical = 10, int subdivisionCircumference = 20, glm::mat4 applyTransform = glm::mat4(1));
	void create_cube(glm::mat4 applyTransform = glm::mat4(1));
	void create_line_cube(glm::mat4 applyTransform = glm::mat4(1));
	void create_grid(bool twoSided = false, int subdivisionsX = 10, int subdivisionsZ = 10, glm::mat4 applyTransform = glm::mat4(1));

	simple_geometry& set_flags(flags aFlags) {
		mFlags = aFlags;
		return *this;
	}

	avk::buffer mPositionsBuffer;
	avk::buffer mIndexBuffer;
	avk::buffer mTexCoordsBuffer;
	avk::buffer mNormalsBuffer;
	avk::buffer mTangentsBuffer;
	avk::buffer mBitangentsBuffer;

private:
	avk::queue* mQueue;
	flags mFlags = flags::none;
	void apply_transform_and_create_buffers(glm::mat4 &applyTransform, std::vector<glm::vec3> &vert, std::vector<uint32_t> &indx);
	void apply_transform_and_create_buffers(glm::mat4 &applyTransform, std::vector<glm::vec3> &vert, std::vector<uint32_t> &indx, std::vector<glm::vec3> &norm, std::vector<glm::vec2> &texc);
	void apply_transform_and_create_buffers(glm::mat4 &applyTransform, std::vector<glm::vec3> &vert, std::vector<uint32_t> &indx, std::vector<glm::vec3> &norm, std::vector<glm::vec2> &texc, std::vector<glm::vec3> &aTangents, std::vector<glm::vec3> &aBitangents);
	std::tuple<std::vector<glm::vec3>,std::vector<glm::vec3>> create_tangents_and_bitangents(const std::vector<glm::vec3> &vert, const std::vector<uint32_t> &indx, const std::vector<glm::vec3> &norm, const std::vector<glm::vec2> &texc);
};


// logical operators for flags:

inline simple_geometry::flags operator| (simple_geometry::flags a, simple_geometry::flags b)
{
	typedef std::underlying_type<simple_geometry::flags>::type EnumType;
	return static_cast<simple_geometry::flags>(static_cast<EnumType>(a) | static_cast<EnumType>(b));
}

inline simple_geometry::flags operator& (simple_geometry::flags a, simple_geometry::flags b)
{
	typedef std::underlying_type<simple_geometry::flags>::type EnumType;
	return static_cast<simple_geometry::flags>(static_cast<EnumType>(a) & static_cast<EnumType>(b));
}

inline simple_geometry::flags& operator |= (simple_geometry::flags& a, simple_geometry::flags b)
{
	return a = a | b;
}

inline simple_geometry::flags& operator &= (simple_geometry::flags& a, simple_geometry::flags b)
{
	return a = a & b;
}

inline simple_geometry::flags exclude(simple_geometry::flags original, simple_geometry::flags toExclude)
{
	typedef std::underlying_type<simple_geometry::flags>::type EnumType;
	return static_cast<simple_geometry::flags>(static_cast<EnumType>(original) & (~static_cast<EnumType>(toExclude)));
}

