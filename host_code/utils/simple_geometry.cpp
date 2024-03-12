#include <context_vulkan.hpp>

#include "simple_geometry.hpp"

void simple_geometry::create_cone(int subdivision, bool closedBase, glm::mat4 applyTransform)
{
	if (mFlags != flags::none) throw avk::runtime_error("Unsupported geometry flags");

	std::vector<glm::vec3> vert;
	std::vector<uint32_t>  indx;

	if (subdivision < 3) subdivision = 3;
	const float y = 1.0f;

	float deltaTheta = glm::radians(360.f) / static_cast<float>(subdivision);

	int numVert = subdivision + closedBase ? 2 : 1;
	vert.reserve(numVert);

	vert.push_back(glm::vec3(0, 0, 0)); // apex
	for (int stepTheta = 0; stepTheta < subdivision; ++stepTheta) {
		float theta = stepTheta * deltaTheta;
		float x =  cos(theta);
		float z = -sin(theta);
		vert.push_back(glm::vec3(x, y, z));
	}
	if (closedBase) vert.push_back(glm::vec3(0, y, 0)); // base center

	for (int i = 0; i < subdivision; ++i) {
		int j = (i + 1) % subdivision;
		indx.push_back(0);
		indx.push_back(1 + j);
		indx.push_back(1 + i);
	}
	if (closedBase) {
		int idxCenter = numVert - 1;

		for (int i = 0; i < subdivision; ++i) {
			int j = (i + 1) % subdivision;
			indx.push_back(idxCenter);
			indx.push_back(1 + i);
			indx.push_back(1 + j);
		}
	}

	apply_transform_and_create_buffers(applyTransform, vert, indx);
}

void simple_geometry::create_sphere(int subdivisionVertical, int subdivisionCircumference, glm::mat4 applyTransform)
{
	const bool explicitTangent = true; // explicitly calc tangents - avoids discontinuity at tex coords seam

	std::vector<glm::vec3> vert;
	std::vector<glm::vec3> norm;
	std::vector<glm::vec3> tang;
	std::vector<glm::vec3> bita;
	std::vector<glm::vec2> texc;
	std::vector<uint32_t>  indx;

	if (subdivisionVertical      < 1) subdivisionVertical      = 1;
	if (subdivisionCircumference < 3) subdivisionCircumference = 3;

	float deltaPhi   = glm::radians(180.f) / static_cast<float>(subdivisionVertical);
	float deltaTheta = glm::radians(360.f) / static_cast<float>(subdivisionCircumference);

	int numVert = (subdivisionVertical + 1) * (subdivisionCircumference + 1);
	vert.reserve(numVert);
	texc.reserve(numVert);
	norm.reserve(numVert);
	tang.reserve(numVert);
	bita.reserve(numVert);

	for (int stepPhi = 0; stepPhi <= subdivisionVertical; ++stepPhi) {
		float phi = stepPhi * deltaPhi;
		float y = cos(phi);
		float r = sin(phi);
		float v = static_cast<float>(stepPhi) / subdivisionVertical;
		for (int stepTheta = 0; stepTheta <= subdivisionCircumference; ++stepTheta) {	// one extra row needed for tex coords
			float theta = stepTheta * deltaTheta;
			float x =  cos(theta) * r;
			float z = -sin(theta) * r;
			float u = static_cast<float>(stepTheta) / subdivisionCircumference;
			vert.push_back(glm::vec3(x, y, z));
			texc.push_back(glm::vec2(u, v));
			norm.push_back(glm::normalize(glm::vec3(x, y, z)));
			if (explicitTangent) {
				glm::mat4 rPhi   = glm::rotate(phi,   glm::vec3(0, 0, -1));
				glm::mat4 rTheta = glm::rotate(theta, glm::vec3(0, 1, 0));
				glm::vec3 t = glm::vec3(rTheta * glm::vec4(0, 0, -1, 0));
				glm::vec3 b = glm::vec3(rTheta * rPhi * glm::vec4(1, 0, 0, 0));
				tang.push_back(t);
				bita.push_back(b);
			}
		}
	}

	// TODO: reserve indx

	for (int lat = 0; lat < subdivisionVertical; ++lat) {
		int v_start = lat * (subdivisionCircumference + 1);
		for (int lon = 0; lon < subdivisionCircumference; ++lon) {
			int a = v_start + lon;
			int b = v_start + lon + (subdivisionCircumference + 1);
			int c = v_start + lon + (subdivisionCircumference + 1) + 1;
			int d = v_start + lon                                  + 1;
			indx.push_back(a);
			indx.push_back(b);
			indx.push_back(c);
			indx.push_back(c);
			indx.push_back(d);
			indx.push_back(a);
		}
	}

	apply_transform_and_create_buffers(applyTransform, vert, indx, norm, texc, tang, bita);
}

void simple_geometry::create_cube(glm::mat4 applyTransform)
{
	if (mFlags != flags::none) throw avk::runtime_error("Unsupported geometry flags");

	const float d = 1.0f;
	std::vector<glm::vec3> vert = {
		{-d, -d,  d}, { d, -d,  d}, { d, -d, -d}, {-d, -d, -d},	// bottom
		{-d,  d,  d}, { d,  d,  d}, { d,  d, -d}, {-d,  d, -d},	// top
	};
	std::vector<uint32_t> indx = {
		0, 1, 5,  0, 5, 4, // +z face
		2, 3, 7,  2, 7, 6, // -z face
		1, 2, 6,  1, 6, 5, // +x face
		3, 0, 4,  3, 4, 7, // -x face
		4, 5, 6,  4, 6, 7, // +y face
		3, 2, 1,  3, 1, 0  // -y face
	};

	apply_transform_and_create_buffers(applyTransform, vert, indx);
}

void simple_geometry::create_line_cube(glm::mat4 applyTransform)
{
	if (mFlags != flags::none) throw avk::runtime_error("Unsupported geometry flags");

	const float d = 1.0f;
	std::vector<glm::vec3> vert = {
		{-d, -d,  d}, { d, -d,  d}, { d, -d, -d}, {-d, -d, -d},	// bottom
		{-d,  d,  d}, { d,  d,  d}, { d,  d, -d}, {-d,  d, -d},	// top
	};
	std::vector<uint32_t> indx = {
		0, 1,  1, 2,  2, 3,  3, 0,
		4, 5,  5, 6,  6, 7,  7, 4,
		0, 4,  1, 5,  2, 6,  3, 7
	};

	apply_transform_and_create_buffers(applyTransform, vert, indx);
}

void simple_geometry::create_grid(bool twoSided, int subdivisionsX, int subdivisionsZ, glm::mat4 applyTransform)
{
	if (mFlags != flags::none) throw avk::runtime_error("Unsupported geometry flags");

	std::vector<glm::vec3> vert;
	std::vector<uint32_t>  indx;

	int m = subdivisionsX + 2;
	int n = subdivisionsZ + 2;
	vert.reserve(n * m);
	indx.reserve((n - 1) * (m - 1) * 6 * (twoSided ? 2 : 1));

	float dx = 1.0f / static_cast<float>(m - 1);
	float dz = 1.0f / static_cast<float>(n - 1);
	for (int zz = 0; zz < n; ++zz) {
		for (int xx = 0; xx < m; ++xx) {
			vert.push_back(glm::vec3(-0.5f + xx * dx, 0, -0.5f + zz * dz));
		}
	}

	for (int zz = 0; zz < n-1; ++zz) {
		for (int xx = 0; xx < m-1; ++xx) {
			uint32_t p0 =  zz      * m + xx;
			uint32_t p1 = (zz + 1) * m + xx;
			uint32_t p2 = (zz + 1) * m + xx + 1;
			uint32_t p3 =  zz      * m + xx + 1;
			indx.push_back(p0);
			indx.push_back(p1);
			indx.push_back(p2);
			indx.push_back(p2);
			indx.push_back(p3);
			indx.push_back(p0);
			if (twoSided) {
				indx.push_back(p3);
				indx.push_back(p2);
				indx.push_back(p1);
				indx.push_back(p1);
				indx.push_back(p0);
				indx.push_back(p3);
			}
		}
	}

	apply_transform_and_create_buffers(applyTransform, vert, indx);
}

void simple_geometry::apply_transform_and_create_buffers(glm::mat4 & applyTransform, std::vector<glm::vec3>& vert, std::vector<uint32_t>& indx) {
	if (mFlags != flags::none) throw avk::runtime_error("Unsupported geometry flags");
	std::vector<glm::vec3> dummyV3;
	std::vector<glm::vec2> dummyV2;
	apply_transform_and_create_buffers(applyTransform, vert, indx, dummyV3, dummyV2, dummyV3, dummyV3);
}

void simple_geometry::apply_transform_and_create_buffers(glm::mat4 &applyTransform, std::vector<glm::vec3> &vert, std::vector<uint32_t> &indx, std::vector<glm::vec3> &norm, std::vector<glm::vec2> &texc) {
	std::vector<glm::vec3> dummyV3;
	apply_transform_and_create_buffers(applyTransform, vert, indx, norm, texc, dummyV3, dummyV3);
}

void simple_geometry::apply_transform_and_create_buffers(glm::mat4 &applyTransform, std::vector<glm::vec3> &vert, std::vector<uint32_t> &indx, std::vector<glm::vec3> &norm, std::vector<glm::vec2> &texc, std::vector<glm::vec3> &aTangents, std::vector<glm::vec3> &aBitangents)
{
	bool bNormals   = (mFlags & flags::normals)   != flags::none;
	bool bTexCoords = (mFlags & flags::texCoords) != flags::none;
	bool bTangents  = (mFlags & flags::tangents)  != flags::none;

	// apply transform
	for (auto &v : vert) v = glm::vec3(applyTransform * glm::vec4(v, 1));
	if (bNormals) {
		glm::mat3 invTransp = glm::inverse(glm::transpose(glm::mat3(applyTransform)));
		for (auto &n : norm) n = invTransp * n;
	}

	// create tangents (after transform)
	std::vector<glm::vec3> tang, bita;
	if (bTangents) {
		if (aTangents.size() == vert.size() && aBitangents.size() == vert.size()) {
			tang = std::move(aTangents);
			bita = std::move(aBitangents);
		} else {
			auto [t, b] = create_tangents_and_bitangents(vert, indx, norm, texc);
			tang = std::move(t);
			bita = std::move(b);
		}
	}


	avk::memory_usage memUsg = avk::memory_usage::device;
	mPositionsBuffer      = avk::context().create_buffer(memUsg, {}, avk::vertex_buffer_meta::create_from_data(vert).describe_only_member(vert[0], avk::content_description::position));
	mIndexBuffer          = avk::context().create_buffer(memUsg, {}, avk::index_buffer_meta ::create_from_data(indx).describe_only_member(indx[0], avk::content_description::index));
	if (bNormals) {
		mNormalsBuffer    = avk::context().create_buffer(memUsg, {}, avk::vertex_buffer_meta::create_from_data(norm).describe_only_member(norm[0], avk::content_description::normal));
	}
	if (bTexCoords) {
		mTexCoordsBuffer  = avk::context().create_buffer(memUsg, {}, avk::vertex_buffer_meta::create_from_data(texc).describe_only_member(texc[0], avk::content_description::texture_coordinate));
	}
	if (bTangents) {
		mTangentsBuffer   = avk::context().create_buffer(memUsg, {}, avk::vertex_buffer_meta::create_from_data(tang).describe_only_member(tang[0], avk::content_description::tangent));
		mBitangentsBuffer = avk::context().create_buffer(memUsg, {}, avk::vertex_buffer_meta::create_from_data(bita).describe_only_member(bita[0], avk::content_description::bitangent));
	}


	// TODO: Do we not have an mQueue here? Add one?!

	//auto fence = avk::context().record_and_submit_with_fence_old_sync_replacement({
	//		mPositionsBuffer->fill(vert.data(), 0),
	//		mIndexBuffer->fill(indx.data(), 0)
	//	});
	//fence->wait_until_signalled();

	// TODO: how can we do something like: "if (bNormals) mNormalsBuffer->fill()" in recording? -> is the solution below ok? - ask JU!

	std::vector<avk::recorded_commands_t> recordedCmds;
	recordedCmds = {
		mPositionsBuffer->fill(vert.data(), 0),
		mIndexBuffer->fill(indx.data(), 0)
	};
	if (bNormals)   recordedCmds.push_back(mNormalsBuffer   ->fill(norm.data(), 0));
	if (bTexCoords) recordedCmds.push_back(mTexCoordsBuffer ->fill(texc.data(), 0));
	if (bTangents)  recordedCmds.push_back(mTangentsBuffer  ->fill(tang.data(), 0));
	if (bTangents)  recordedCmds.push_back(mBitangentsBuffer->fill(bita.data(), 0));
	auto fence = avk::context().record_and_submit_with_fence(recordedCmds, *mQueue);
	fence->wait_until_signalled();
}

std::tuple<std::vector<glm::vec3>, std::vector<glm::vec3>> simple_geometry::create_tangents_and_bitangents(const std::vector<glm::vec3>& vert, const std::vector<uint32_t>& indx, const std::vector<glm::vec3>& norm, const std::vector<glm::vec2>& texc)
{
	// loosely based on: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/#computing-the-tangents-and-bitangents
	// and on own code from CGUE :)

	if (indx.size() % 3 != 0) throw avk::runtime_error("simple_geometry::create_tangents_and_bitangents: not a triangle mesh?");

	size_t numv = vert.size();
	std::vector<glm::vec3> t(numv, glm::vec3(0));
	std::vector<glm::vec3> b(numv, glm::vec3(0));
	std::vector<int> hitCount(numv, 0);

	// process each face
	size_t numFaces = indx.size() / 3;
	for (size_t face = 0; face < numFaces; ++face) {
		auto &i0 = indx[3 * face + 0];
		auto &i1 = indx[3 * face + 1];
		auto &i2 = indx[3 * face + 2];

		auto &v0 = vert[i0];
		auto &v1 = vert[i1];
		auto &v2 = vert[i2];

		auto &uv0 = texc[i0];
		auto &uv1 = texc[i1];
		auto &uv2 = texc[i2];

		glm::vec3 dPos1 = v1 - v0;
		glm::vec3 dPos2 = v2 - v0;
		glm::vec2 dUV1  = uv1 - uv0;
		glm::vec2 dUV2  = uv2 - uv0;

		float r = 1.0f / (dUV1.x * dUV2.y - dUV1.y * dUV2.x);
		glm::vec3 tangent   = (dPos1 * dUV2.y - dPos2 * dUV1.y) * r;
		glm::vec3 bitangent = (dPos2 * dUV1.x - dPos1 * dUV2.x) * r;

		// add upp all tangents/bitangents per vertex
		t[i0] += tangent; b[i0] += bitangent; hitCount[i0]++;
		t[i1] += tangent; b[i1] += bitangent; hitCount[i1]++;
		t[i2] += tangent; b[i2] += bitangent; hitCount[i2]++;
	}

	// average tangents/bitangents per vertex
	for (size_t i = 0; i < numv; ++i) {
		int hits = hitCount[i];
		if (hits) {
			t[i] /= static_cast<float>(hits);
			b[i] /= static_cast<float>(hits);
		}
	}

	return std::make_tuple(std::move(t), std::move(b));
}


