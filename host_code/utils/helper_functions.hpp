#pragma once

#include <serializer.hpp>

#include "material_image_helpers.hpp"
#include "model.hpp"
#include "orca_scene.hpp"
#include "../../shaders/lightsource_limits.h"

namespace helpers
{
	/** A small helper struct which contains data for a draw call,
	 *	including all relevant vertex attributes, and the material index.
	 */
	struct data_for_draw_call
	{
		avk::buffer mIndexBuffer;
		avk::buffer mPositionsBuffer;
		avk::buffer mTexCoordsBuffer;
		avk::buffer mNormalsBuffer;
		avk::buffer mTangentsBuffer;
		avk::buffer mBitangentsBuffer;
		int mMaterialIndex;
		glm::mat4 mModelMatrix;
	};

	static bool contains_blue_curtains(std::vector<std::tuple<const avk::model_t&, std::vector<avk::mesh_index_t>>>& aSelectedModelsAndMeshes)
	{
		size_t a = 0;
		for (; a < aSelectedModelsAndMeshes.size(); ++a) {
			auto& tpl = aSelectedModelsAndMeshes[a];
			const auto& model = std::get<const avk::model_t&>(tpl);
			if (model.path().find("fabric") == std::string::npos) {
				continue;
			}
			std::vector<size_t>& meshIndices = std::get<std::vector<size_t>>(tpl);
			for (size_t i = 0; i < meshIndices.size(); ++i) {
				auto meshName = model.name_of_mesh(meshIndices[i]);

				// Exclude one blue curtain (of a total of three) by modifying the indices before uploading them to a GPU buffer:
				if (meshName == "sponza_326") {
					return true;
				}
			}
		}
		return false;
	}

	static void set_terrain_material_config(avk::orca_scene_t& aScene)
	{
		auto applyMaterialChanges = [](avk::material_config &m, bool isTerrain) {
			m.mAmbientReflectivity	       = glm::vec4{ 1.0f, 1.0f, 1.0f, 1.0f };
			m.mDiffuseReflectivity	       = glm::vec4{ 1.0f, 1.0f, 1.0f, 1.0f };
			m.mSpecularReflectivity	       = glm::vec4{ 0.0f, 0.0f, 0.0f, 0.0f };
			m.mEmissiveColor		       = glm::vec4{ 0.0f, 0.0f, 0.0f, 0.0f };
			m.mShininess			       = 100.0f; // <- this is different in model file (10.0f)
			m.mHeightTex			       = "assets/terrain/large_metal_debris_Displacement.jpg"; // assimp cannot load height/displacement textures from .dae -> add it manually
			m.mHeightTexOffsetTiling       = isTerrain ? glm::vec4{ 0.0f, 0.0f, 32.0f, 32.0f } : glm::vec4{ 0.0f, 0.0f, 10.0f, 10.0f };
			m.mHeightTexBorderHandlingMode = { avk::border_handling_mode::repeat, avk::border_handling_mode::repeat }; // already set correctly for the other textures
		};

		// Select the terrain models/meshes
		auto terrainModelIndices = aScene.select_models([](size_t index, const avk::model_data& modelData){
			return std::string::npos != modelData.mName.find("terrain") || std::string::npos != modelData.mName.find("debris");
			});

		// Assign the material config to all meshes
		for (auto i : terrainModelIndices) {
			bool isTerrain = (std::string::npos != aScene.model_at_index(i).mName.find("terrain"));
			auto meshes = aScene.model_at_index(i).mLoadedModel->select_all_meshes();
			for (auto j : meshes) {
				auto m = aScene.model_at_index(i).mLoadedModel->material_config_for_mesh(j);
				applyMaterialChanges(m, isTerrain);
				aScene.model_at_index(i).mLoadedModel->set_material_config_for_mesh(j, m);
			}
		}
	}


	// We're only going to tessellate terrain materials. Set the tessellation factor for those to 1.
	// Indicate that the other materials shall not be tessellated/displaced with a tessellation factor of 0.
	static void enable_tessellation_for_specific_meshes(avk::orca_scene_t& aScene)
	{
		for (auto& model : aScene.models()) {
			const bool isToBeTessellated = std::string::npos != model.mName.find("terrain") || std::string::npos != model.mName.find("debris");
			auto meshIndices = model.mLoadedModel->select_all_meshes();
			for (auto i : meshIndices) {
				auto m = model.mLoadedModel->material_config_for_mesh(i);
				m.mCustomData[0] = isToBeTessellated ? 1.0f : 0.0f;
				model.mLoadedModel->set_material_config_for_mesh(i, m);
			}
		}
	}

	// makes only sense for meshes that are to be tessellated
	static void set_mesh_specific_displacement_strength(avk::orca_scene_t& aScene)
	{
		for (auto& model : aScene.models()) {
			// Compute a displacement strength that fits to the normal map strength:

			// Displacement distance in relation to the texel size - texture specific value
			auto displacementInTexels = 400.0f;

			// Average size of u and v in object space (actually, mesh specific value)
			auto uvScaleOS = 200.0f;
			if (std::string::npos != model.mName.find("terrain")) {
				uvScaleOS = 2040.0f;
			}
			if (std::string::npos != model.mName.find("debris")) {
				uvScaleOS = 1200.0f;
			}

			auto meshIndices = model.mLoadedModel->select_all_meshes();
			for (auto i : meshIndices) {
				auto m = model.mLoadedModel->material_config_for_mesh(i);
				const bool isToBeTessellated = m.mCustomData[0] != 0.0f;
				if (!isToBeTessellated) {
					continue;
				}

				// Compute approximate size of a texel in object space, which depends on the
				// average size of u and v in object space, the texture's size and tiling.

				int width = 1024, height = 1024, comp = 4; // just init with something if stbi_info fails
				stbi_info(m.mHeightTex.c_str(), &width, &height, &comp);

				auto tiling = m.mHeightTexOffsetTiling[2];

				auto texelSizeOS = uvScaleOS / (tiling * width);

				// Compute the displacement strength factor for this mesh in object space
				// (actually, transform m_displacement_strength from "texture space" to object space)
				float displacementStrengthFactorOS = displacementInTexels * texelSizeOS;

				m.mCustomData[1] = displacementStrengthFactorOS;
				model.mLoadedModel->set_material_config_for_mesh(i, m);
			}
		}
	}

	// Increase the specularity of some submeshes so that they get reflections applied more strongly
	static void increase_specularity_of_some_submeshes(avk::orca_scene_t& aScene)
	{
		auto sponzaStructureIndex = aScene.select_models([](size_t index, const avk::model_data& modelData) {
			return std::string::npos != modelData.mName.find("sponza_structure");
		});

		assert(sponzaStructureIndex.size() == 1);

		// Assign the material config to the "floor" and "lion" meshes
		auto& model = aScene.model_at_index(sponzaStructureIndex[0]).mLoadedModel;
		auto meshes = model->select_meshes([&model](size_t meshIndex, const aiMesh*) {
			return 0 == model->name_of_mesh(meshIndex).find("floor") || 0 == model->name_of_mesh(meshIndex).find("lion");
			});
		for (auto j : meshes) {
			auto mat = model->material_config_for_mesh(j);
			mat.mReflectiveColor = glm::vec4{ 0.9f };
			mat.mCustomData[2] = 0.75f; // Set a normal mapping strength decrease factor
			model->set_material_config_for_mesh(j, mat);
			auto matafterarsch = model->material_config_for_mesh(j);
			auto matafterarsc2h = model->material_config_for_mesh(j);
		}
	}

	// assign additional PBS materials to Sponza
	static void setup_sponza_pbs_materials(avk::orca_scene_t& aScene) {
		// In the shaders, we can use
		//   metallic  = material.mMetallic  * sample from reflection texture
		//   roughness = material.mRoughness * sample from extra texture
		// both textures default to white if unset

		const std::string pbsTexturePath = "assets/sponza_pbr_textures/";
		struct PbsData {
			std::string modelName;
			std::string materialName;
			std::string roughnessTextureName;
			std::string metallicTextureName;
		};
		std::vector<PbsData> pbsData = {
			{"sponza_structure",	"arch",				"Sponza_Arch_roughness.png",			"Dielectric_metallic.png"},
			{"sponza_structure",	"bricks",			"Sponza_Bricks_a_Roughness.png",		"Dielectric_metallic.png"},
			//{"sponza_structure",	"bricks_NONE",		"",			""},
			{"sponza_structure",	"ceiling",			"Sponza_Ceiling_roughness.png",			"Dielectric_metallic.png"},
			{"sponza_structure",	"column_a",			"Sponza_Column_a_roughness.png",		"Dielectric_metallic.png"},
			{"sponza_structure",	"column_b",			"Sponza_Column_b_roughness.png",		"Dielectric_metallic.png"},
			{"sponza_structure",	"column_c",			"Sponza_Column_c_roughness.png",		"Dielectric_metallic.png"},
			{"sponza_structure",	"details",			"Sponza_Details_roughness.png",			"Dielectric_metallic.png"},
			{"sponza_structure",	"flagpole",			"Sponza_FlagPole_roughness.png",		"Metallic_metallic.png"},
			{"sponza_structure",	"floor",			"Sponza_Floor_roughness.png",			"Dielectric_metallic.png"},
			{"sponza_structure",	"roof",				"Sponza_Roof_roughness.png",			"Dielectric_metallic.png"},
			{"sponza_structure",	"vase",				"Vase_roughness.png",					"Dielectric_metallic.png"},
			{"sponza_structure",	"Material__25",		"Lion_Roughness.png",					"Dielectric_metallic.png"},
			{"sponza_structure",	"Material__298",	"Background_roughness.png",				"Dielectric_metallic.png"},
			{"sponza_fabric",		"fabric_a",			"Sponza_Fabric_roughness.png",			"Sponza_Fabric_metallic.png"},
			{"sponza_fabric",		"fabric_c",			"Sponza_Curtain_roughness.png",			"Sponza_Curtain_metallic.png"},
			{"sponza_fabric",		"fabric_d",			"Sponza_Fabric_roughness.png",			"Sponza_Fabric_metallic.png"},
			{"sponza_fabric",		"fabric_e",			"Sponza_Fabric_roughness.png",			"Sponza_Fabric_metallic.png"},
			{"sponza_fabric",		"fabric_f",			"Sponza_Curtain_roughness.png",			"Sponza_Curtain_metallic.png"},
			{"sponza_fabric",		"fabric_g",			"Sponza_Curtain_roughness.png",			"Sponza_Curtain_metallic.png"},
			{"sponza_plants",		"chain",			"ChainTexture_Roughness.png",			"ChainTexture_Metallic.png"},
			{"sponza_plants",		"leaf",				"Sponza_Thorn_roughness.png",			"Dielectric_metallic.png"},
			{"sponza_plants",		"vase_hanging",		"VaseHanging_roughness.png",			"Metallic_metallic.png"},
			{"sponza_plants",		"vase_round",		"VaseRound_roughness.png",				"Dielectric_metallic.png"},
			{"sponza_plants",		"Material__57",		"VasePlant_roughness.png",				"Dielectric_metallic.png"},
			//{"sponza_debris",		"None",				"",			""},
			//{"surrounding_terrain",	"None",				"",			""},
			// TODO: roughness for debris/terrain ?
		};

		for (auto& modelData: aScene.models()) {
			auto meshIndices = modelData.mLoadedModel->select_all_meshes();
			for (auto iMesh: meshIndices) {
				bool handled = false;
				auto mat = modelData.mLoadedModel->material_config_for_mesh(iMesh);
				for (auto& pbs : pbsData) {
					if (modelData.mName == pbs.modelName && mat.mName == pbs.materialName) {
						// make sure the material has a diffuse texture, as we use its offset tiling and border handling mode
						bool ok = true;
						if (mat.mDiffuseTex == "") {
							printf("No diffuse texture??\n");
							ok = false;
						}
						if (ok) {
							mat.mReflectionTex						= pbsTexturePath + pbs.metallicTextureName;
							mat.mReflectionTexBorderHandlingMode	= mat.mDiffuseTexBorderHandlingMode;
							mat.mReflectionTexOffsetTiling			= mat.mDiffuseTexOffsetTiling;
							mat.mReflectionTexRotation				= mat.mDiffuseTexRotation;
							mat.mReflectionTexUvSet					= mat.mDiffuseTexUvSet;

							mat.mExtraTex							= pbsTexturePath + pbs.roughnessTextureName;
							mat.mExtraTexBorderHandlingMode			= mat.mDiffuseTexBorderHandlingMode;
							mat.mExtraTexOffsetTiling				= mat.mDiffuseTexOffsetTiling;
							mat.mExtraTexRotation					= mat.mDiffuseTexRotation;
							mat.mExtraTexUvSet						= mat.mDiffuseTexUvSet;

							mat.mMetallic  = 1.0f;
							mat.mRoughness = 1.0f;
							modelData.mLoadedModel->set_material_config_for_mesh(iMesh, mat);
							handled = true;
						}
					} else if (modelData.mName == "sponza_debris" || modelData.mName == "surrounding_terrain") {
						// special handling - these already have a metallic texture (but no roughness)
						mat.mMetallic  = 1.0f;
						mat.mRoughness = 0.5f;
						modelData.mLoadedModel->set_material_config_for_mesh(iMesh, mat);
						handled = true;
					}
				}
				if (!handled) {
					printf("- No PBS info for model \"%s\", mesh #%d, material \"%s\"\n", modelData.mName.c_str(), int(iMesh), mat.mName.c_str());
				}
			}
		}
	}

	// identify assignment 3 IBL model
	static int identify_a3_special_ibl_model(std::vector<std::tuple<const avk::model_t&, std::vector<avk::mesh_index_t>>>& aSelectedModelsAndMeshes) {
		size_t a = 0;
		for (; a < aSelectedModelsAndMeshes.size(); ++a) {
			auto& tpl = aSelectedModelsAndMeshes[a];
			const avk::model_t& model = std::get<const avk::model_t&>(tpl);
			if (model.path().find("sponza_structure") == std::string::npos) {
				continue;
			}
			std::vector<size_t>& meshIndices = std::get<std::vector<size_t>>(tpl);
			for (size_t i = 0; i < meshIndices.size(); ++i) {
				auto meshName = model.name_of_mesh(meshIndices[i]);
				if (meshName == "vase_376_sponza_376") {
					return 1;
				}
			}
		}
		return 0;
	}

	// create index data for assignment 3 IBL model
	static void create_a3_special_ibl_model_indices_and_modelmatrix(int aSpecialModelId, const std::vector<uint32_t>& aOrigIndices, std::vector<uint32_t>& aNewIndices, glm::mat4& aModelMatrix) {
		if (aSpecialModelId == 1) {
			assert(aOrigIndices.size() % (4*3) == 0); // 4 vases
			aNewIndices.clear();
			aNewIndices.insert(aNewIndices.begin(), aOrigIndices.begin(), aOrigIndices.begin() + aOrigIndices.size() / 4);
			// center of untransformed first vase is 1119.6947,66.161775,-449.02377
			aModelMatrix =  glm::scale(glm::vec3(0.01f)) * glm::translate(glm::vec3(-1119.6947, -66.161775, 449.02377));
		}
	}

	// add an extra material for rendering IBL spheres in assignment 3
	static void add_extra_material_for_a3_ibl(std::vector<avk::material_config>& allMaterials) {
		// simply red
		avk::material_config mat;
		mat.mName = "a3_ibl_spheres";
		mat.mDiffuseReflectivity = { 1.0f, 0.0f, 0.0f, 0.0f };
		mat.mMetallic  = 1.0f;
		mat.mRoughness = 1.0f;
		allMaterials.push_back(mat);
	}



	/**	Load an ORCA scene from file
	 *
	 */
	static std::tuple<
		     avk::buffer, std::vector<avk::image_sampler>, std::vector<data_for_draw_call>
	       >
		   load_models_and_scenes_from_file(std::vector<std::tuple<std::string, glm::mat4>> aPathsAndTransforms, avk::queue* aQueue)
	{
		const auto cacheFilePath = std::accumulate(
			std::begin(aPathsAndTransforms), std::end(aPathsAndTransforms),
			std::string{ "a1" },
			[](const auto& a, const auto& b) { return a + "_" + avk::extract_file_name(std::get<std::string>(b)); }
		) + ".cache";
		// If a cache file exists, i.e. the scene was serialized during a previous load, initialize the serializer in deserialize mode,
		// else initialize the serializer in serialize mode to create the cache file while processing the scene.
		auto serializer = avk::serializer(cacheFilePath, avk::does_cache_file_exist(cacheFilePath) 
			? avk::serializer::mode::deserialize
			: avk::serializer::mode::serialize
		);

		if (serializer.mode() == avk::serializer::mode::serialize) {
			for (auto &pt : aPathsAndTransforms) {
				LOG_INFO(std::format("About to load 3D model/scene from {}", avk::extract_file_name(std::get<std::string>(pt))));
			}
			LOG_INFO("Please be patient, this might take a while...");
		}
		else {
			LOG_INFO(std::format("About to load cached 3D model/scene from {}", cacheFilePath));
		}

		// The following loop gathers all the vertex and index data PER MATERIAL and constructs the buffers and materials.
		// Later, we'll use ONE draw call PER MATERIAL to draw the whole scene.
		std::vector<avk::material_config> materialConfigs;
		size_t materialIndex = 0;
		std::vector<data_for_draw_call> drawCalls;

		size_t numLoadees = (serializer.mode() == avk::serializer::mode::serialize) ? aPathsAndTransforms.size() : 0;
		serializer.archive(numLoadees);
		assert(numLoadees == aPathsAndTransforms.size());

		for (size_t l = 0; l < numLoadees; ++l) {
			const auto& [path, transform] = aPathsAndTransforms[l];
			// Load an ORCA scene from file:
			avk::orca_scene orca;
			avk::model_data model;
			std::unordered_map<avk::material_config, std::vector<avk::model_and_mesh_indices>> distinctMaterialsFromFile;
			std::function<avk::model_data& (avk::model_index_t)> getModelData;
			
			// Load orca scene for usage and serialization, loading the scene is not required if a cache file exists, i.e. mode == deserialize
			if (serializer.mode() == avk::serializer::mode::serialize) {
				int triesLeft = 2;
				bool tryToLoadAsModel = !path.ends_with(".fscene"); // if it ends with .fscene we can be pretty sure it is a scene - so try that first!
				bool succeeded = false;
				while (!succeeded && (triesLeft > 0)) {
					try {
						if (tryToLoadAsModel) {
							model.mFileName = path;
							model.mName = path;
							model.mInstances = { avk::model_instance_data{ path, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{1.f, 1.f, 1.f}, glm::vec3{0.f, 0.f, 0.f} } };
							model.mFullPathName = path;
							model.mLoadedModel = avk::model_t::load_from_file(path, aiProcess_PreTransformVertices | aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);
							// Get all the different materials from the model:
							auto fromModel = model.mLoadedModel->distinct_material_configs(true);
							for (auto& [matConfig, meshIndices] : fromModel) {
								distinctMaterialsFromFile[matConfig].emplace_back(0, std::move(meshIndices));
							}
							getModelData = [&](avk::model_index_t aIndex) -> avk::model_data& { return model; };
						} else {
							//! ATTN: orca_scene_t::load_from_file() crashes instead of failing gracefully if path is not an orca file!!
							orca = avk::orca_scene_t::load_from_file(path, aiProcess_PreTransformVertices | aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);
							// Get all the different materials from the whole scene:
							distinctMaterialsFromFile = orca->distinct_material_configs_for_all_models();
							getModelData = [&](avk::model_index_t aIndex) -> avk::model_data& { return orca->model_at_index(aIndex); };
						}
						succeeded = true;
					}
					catch (avk::runtime_error& err) {
						LOG_INFO(std::format("{} is not {} file, failed with error: {}", path, tryToLoadAsModel ? "a model" : "an ORCA", err.what()));
					}
					if (!succeeded) {
						triesLeft--;
						tryToLoadAsModel = !tryToLoadAsModel;
					}
				}
				if (!succeeded) {
					throw avk::runtime_error(std::format("{} is neither a model nor an ORCA file, failed to load.", path));
				}
			}

			// Get number of distinc materials from orca scene and serialize it or retrieve the number from the serializer
			size_t numDistinctMaterials = (serializer.mode() == avk::serializer::mode::serialize) ? distinctMaterialsFromFile.size() : 0;
			serializer.archive(numDistinctMaterials);

			std::vector<avk::recorded_commands_t> commandsToBeExcecuted;

			//auto model = avk::model_t::load_from_file(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);
			//model->distinct_material_configs();
			auto materialFromFile = distinctMaterialsFromFile.begin();
			for (int materialIndexFile = 0; materialIndexFile < numDistinctMaterials; ++materialIndexFile, ++materialIndex) {
				// meshIndices is only needed during serialization, otherwise the serializer handles everything
				// in the respective *_cached functions and meshIndices may be empty when passed to the
				// respective functions.
				size_t numMeshIndices;
				std::vector<avk::model_and_mesh_indices> meshIndices;
				if (serializer.mode() == avk::serializer::mode::serialize) {
					materialConfigs.push_back(materialFromFile->first);
					meshIndices = materialFromFile->second;
					numMeshIndices = materialFromFile->second.size();
					materialFromFile = std::next(materialFromFile);
				}
				// Serialize or retrieve the number of model_and_mesh_indices for the material
				serializer.archive(numMeshIndices);

				// The data in distinctMaterialsOrca encompasses all of the ORCA scene's models.
				for (int meshIndicesIndex = 0; meshIndicesIndex < numMeshIndices; ++meshIndicesIndex) {
					// Convinience function to retrieve the model data via the mesh indices from the orca scene while in serialize mode

					// modelAndMeshes is only needed during serialization, otherwise the following buffers are filled by the
					// serializer from the cache file in the repsective *_cached functions and modelAndMeshes may be empty.
					std::vector<std::tuple<const avk::model_t&, std::vector<avk::mesh_index_t>>> modelAndMeshes;

					std::vector<uint32_t> indices;
					std::vector<glm::vec3> positions;

					if (serializer.mode() == avk::serializer::mode::serialize) {
						modelAndMeshes = avk::make_model_references_and_mesh_indices_selection(getModelData(meshIndices[meshIndicesIndex].mModelIndex).mLoadedModel, meshIndices[meshIndicesIndex].mMeshIndices);
						indices = getModelData(meshIndices[meshIndicesIndex].mModelIndex).mLoadedModel->indices_for_meshes<uint32_t>(meshIndices[meshIndicesIndex].mMeshIndices);
						positions = getModelData(meshIndices[meshIndicesIndex].mModelIndex).mLoadedModel->positions_for_meshes(meshIndices[meshIndicesIndex].mMeshIndices);
						if (contains_blue_curtains(modelAndMeshes)) {
							indices.erase(std::begin(indices), std::begin(indices) + 3 * 4864);
						}
					}

					auto [indicesBuffer,   cmdsIndices  ] = avk::create_buffer_cached<decltype(indices  ), avk::index_buffer_meta >(serializer, indices, avk::content_description::index);
					commandsToBeExcecuted.push_back(std::move(cmdsIndices));
					auto [positionsBuffer, cmdsPositions] = avk::create_buffer_cached<decltype(positions), avk::vertex_buffer_meta>(serializer, positions, avk::content_description::position);
					commandsToBeExcecuted.push_back(std::move(cmdsPositions));

															 // Get a buffer containing all texture coordinates for all submeshes with this material
					auto [texCoordsBuffer, cmdsTexCoordsBuffer] = avk::create_2d_texture_coordinates_flipped_buffer_cached<avk::vertex_buffer_meta>(
						serializer, modelAndMeshes, {}
					);
					commandsToBeExcecuted.push_back(std::move(cmdsTexCoordsBuffer));

															   // Get a buffer containing all normals for all submeshes with this material
					auto [normalsBuffer, cmdsNormalsBuffer] = avk::create_normals_buffer_cached<avk::vertex_buffer_meta>(
						serializer, modelAndMeshes, {}
					);
					commandsToBeExcecuted.push_back(std::move(cmdsNormalsBuffer));

															   // Get a buffer containing all tangents for all submeshes with this material
					auto [tangentsBuffer, cmdsTangentsBuffer] = avk::create_tangents_buffer_cached<avk::vertex_buffer_meta>(
						serializer, modelAndMeshes, {}
					);
					commandsToBeExcecuted.push_back(std::move(cmdsTangentsBuffer));

															   // Get a buffer containing all bitangents for all submeshes with this material
					auto [bitangentsBuffer, cmdsBitangentsBuffer] = avk::create_bitangents_buffer_cached<avk::vertex_buffer_meta>(
						serializer, modelAndMeshes, {}
					);
					commandsToBeExcecuted.push_back(std::move(cmdsBitangentsBuffer));

					 // Get the number of instances from the model and serialize it or retrieve it from the serializer
					size_t numInstances = (serializer.mode() == avk::serializer::mode::serialize) ? getModelData(meshIndices[meshIndicesIndex].mModelIndex).mInstances.size() : 0;
					serializer.archive(numInstances);

					// Create a draw calls for instances with the current material
					for (int instanceIndex = 0; instanceIndex < numInstances; ++instanceIndex) {
						auto& newElement = drawCalls.emplace_back();
						
						newElement.mIndexBuffer       = indicesBuffer;
						newElement.mPositionsBuffer   = positionsBuffer;
						newElement.mNormalsBuffer     = normalsBuffer;
						newElement.mTangentsBuffer    = tangentsBuffer;
						newElement.mBitangentsBuffer  = bitangentsBuffer;
						newElement.mTexCoordsBuffer   = texCoordsBuffer;

						newElement.mMaterialIndex     = static_cast<int>(materialIndex);

						// Create model matrix of instance and serialize it or retrieve it from the serializer
						if (serializer.mode() == avk::serializer::mode::serialize) {
							auto instances = getModelData(meshIndices[meshIndicesIndex].mModelIndex).mInstances;
							newElement.mModelMatrix = avk::matrix_from_transforms(
								instances[instanceIndex].mTranslation, glm::quat(instances[instanceIndex].mRotation), instances[instanceIndex].mScaling
							);
						}
						serializer.archive(newElement.mModelMatrix);

						newElement.mModelMatrix = transform * newElement.mModelMatrix;
					}
				}
			}

			avk::context().record_and_submit_with_fence(std::move(commandsToBeExcecuted), *aQueue)->wait_until_signalled();
		}

		// Convert the materials that were gathered above into a GPU-compatible format, and upload into a GPU storage buffer:
		auto [gpuMaterials, imageSamplers, materialCommands] = avk::convert_for_gpu_usage_cached<avk::material_gpu_data>(
			serializer,
			materialConfigs, true, false,
			avk::image_usage::general_texture,
			avk::filter_mode::anisotropic_16x
		);

		auto materialsBuffer = avk::context().create_buffer(
			avk::memory_usage::device, {},
			avk::storage_buffer_meta::create_from_data(gpuMaterials)
		);

		auto fen = avk::context().record_and_submit_with_fence({
			std::move(materialCommands),
			materialsBuffer->fill(gpuMaterials.data(), 0)
		}, *aQueue);
		fen->wait_until_signalled();

		return std::make_tuple(
			std::move(materialsBuffer), std::move(imageSamplers), std::move(drawCalls)
		);
	}

	const glm::vec3 kInitialPositionOfFirstPointLight  = glm::vec3(-0.64f, 0.45f, 3.35f);
	const glm::vec3 kInitialPositionOfSecondPointLight = glm::vec3(-0.05f, 2.12f, 0.53f);

	static std::vector<avk::lightsource>& get_lights()
	{
		static std::vector<avk::lightsource> sLightsources = [](){
			std::vector<avk::lightsource> ls;

			// Ambient light:
			ls.push_back(avk::lightsource::create_ambient(glm::vec3{1.0f/255.0f, 2.0f/255.0f, 3.0f/255.0f} * 0.5f, "ambient light"));

			// Directional light:
			ls.push_back(avk::lightsource::create_directional(glm::vec3(-0.38f, -0.78f, 0.0f), glm::vec3{13.0f/255.0f, 17.0f/255.0f, 27.0f/255.0f} * 4.0f, "directional light"));

			std::vector<glm::vec3> lightColors;
			lightColors.emplace_back(1.0f, 1.0f, 1.0f);
			lightColors.emplace_back(0.878f, 1.000f, 1.000f);
			lightColors.emplace_back(0.957f, 0.643f, 0.376f);
			lightColors.emplace_back(0.000f, 0.000f, 1.000f);
			lightColors.emplace_back(0.251f, 0.878f, 0.816f);
			lightColors.emplace_back(0.000f, 0.980f, 0.604f);
			lightColors.emplace_back(0.545f, 0.000f, 0.545f);
			lightColors.emplace_back(1.000f, 0.000f, 1.000f);
			lightColors.emplace_back(0.984f, 1.000f, 0.729f);
			lightColors.emplace_back(0.780f, 0.082f, 0.522f);
			lightColors.emplace_back(1.000f, 0.843f, 0.000f);
			lightColors.emplace_back(0.863f, 0.078f, 0.235f);
			lightColors.emplace_back(0.902f, 0.902f, 0.980f);
			lightColors.emplace_back(0.678f, 1.000f, 0.184f);

			std::default_random_engine generator;
			generator.seed(186);
			std::uniform_int_distribution<size_t> distribution(0, lightColors.size() - 1); // generates number in the range 0..light_colors.size()-1

			// Create a light near the walkthrough
			ls.push_back(avk::lightsource::create_pointlight({-0.64f, 0.45f, 3.35f}, lightColors[distribution(generator)] * 3.0f, "pointlight near walkthrough").set_attenuation(1.0f, 0.0f, 5.0f));

			// A1:  Create a (moving) light which is especially useful for evaluating Bonus Task 1
			ls.push_back(avk::lightsource::create_pointlight({-0.05f, 2.12f, 0.53f}, lightColors[distribution(generator)] * 3.0f, "pointlight near parallelepiped").set_attenuation(1.0f, 0.0f, 5.0f));

			{ // Create lots of small lights near the floor
				const auto lbX = -14.2f;
				const auto lbZ = -6.37f;
				const auto nx = 13;
				const auto nz = 6;
				const auto stepX = (12.93f - lbX) / (nx - 1);
				const auto stepZ = (5.65f - lbZ) / (nz - 1);
				for (auto x = 0; x < nx; ++x) {
					for (auto z = 0; z < nz; ++z) {
						ls.push_back(avk::lightsource::create_pointlight(glm::vec3(lbX + x * stepX, 0.1f, lbZ + z * stepZ), lightColors[distribution(generator)]).set_attenuation(1.0f, 0.0f, 30.0f));
					}
				}
			}

			{	// Create several larger lights near the ceiling
				const auto lbX = -13.36f;
				const auto lbZ = -5.46f;
				const auto nx = 6;
				const auto nz = 3;
				const auto stepX = (12.1f - lbX) / (nx - 1);
				const auto stepZ = (4.84f - lbZ) / (nz - 1);
				for (auto x = 0; x < nx; ++x) {
					for (auto z = 0; z < nz; ++z) {
						ls.push_back(avk::lightsource::create_pointlight(glm::vec3(lbX + x * stepX, 7.0f, lbZ + z * stepZ), lightColors[distribution(generator)], std::format("pointlight[{}|{}]", x, z)).set_attenuation(1.0f, 0.0f, 5.666f));
					}
				}
			}

			// create extra point lights if requested per lightsource_limits.h
			constexpr float dAngle = glm::two_pi<float>() / std::max(1, EXTRA_POINTLIGHTS);
			const float radius = 20.0f;
			const float height = 30.0f;
			for (int i = 0; i < EXTRA_POINTLIGHTS; ++i) {
				auto pos = glm::vec3(radius * cos(i * dAngle), height, radius * sin(i * dAngle));
				auto col = glm::vec3(0, 0, 0);
				ls.push_back(avk::lightsource::create_pointlight(pos, col, std::format("extrapointlight[{}]", i)).set_attenuation(1.0f, 0.0f, 5.666f));
			}


			return ls;
		}();
		return sLightsources;
	}

	static void animate_lights(std::vector<avk::lightsource>& aLightsources, float aElapsedTime)
	{
		{
			const auto it = std::find_if(std::begin(aLightsources), std::end(aLightsources), [](const avk::lightsource& ls) { return "pointlight near walkthrough" == ls.mName; });
			if (std::end(aLightsources) != it) {
				const auto speedXZ = 0.5f;
				const auto radiusXZ = 1.5f;
				it->mPosition = glm::vec3{-0.64f, 0.45f, 3.35f} + glm::vec3(radiusXZ * glm::sin(speedXZ * aElapsedTime), 0.0f, radiusXZ * glm::cos(speedXZ * aElapsedTime));
			}
		}
		{
			const auto it = std::find_if(std::begin(aLightsources), std::end(aLightsources), [](const avk::lightsource& ls) { return "pointlight near parallelepiped" == ls.mName; });
			if (std::end(aLightsources) != it) {
				const auto kSpeed = 0.6f;
				const auto kDistanceX = -0.23f;
				const auto kDistanceY = 1.0f;
				it->mPosition = glm::vec3{-0.05f, 2.12f, 0.53f} + glm::vec3(kDistanceX * glm::sin(kSpeed * aElapsedTime), kDistanceY * glm::sin(kSpeed * aElapsedTime), 0.0f);
			}
		}
		{
			const auto it = std::find_if(std::begin(aLightsources), std::end(aLightsources), [](const avk::lightsource& ls) { return "pointlight outside above terrain" == ls.mName; });
			if (std::end(aLightsources) != it) {
				const auto speedXZ = 0.75f;
				const auto radiusXZ = 4.0f;
				it->mPosition = glm::vec3{-2.0f, 1.45f, 17.0f} + glm::vec3(radiusXZ * glm::sin(speedXZ * aElapsedTime), 0.0f, radiusXZ * glm::cos(speedXZ * aElapsedTime));
			}
		}
	}

	static uint32_t get_lightsource_type_begin_index(std::vector<avk::lightsource>& aLightsources, avk::lightsource_type aLightsourceType)
	{
		auto it = std::lower_bound(std::begin(aLightsources), std::end(aLightsources), aLightsourceType, [](const avk::lightsource& a, const avk::lightsource_type& b){
			typedef std::underlying_type<avk::lightsource_type>::type EnumType;
			return static_cast<EnumType>(a.mType) < static_cast<EnumType>(b);
		});
		return static_cast<uint32_t>(std::distance(std::begin(aLightsources), it));
	}

	static uint32_t get_lightsource_type_begin_index(avk::lightsource_type aLightsourceType)
	{
		return get_lightsource_type_begin_index(get_lights(), aLightsourceType);
	}

	static uint32_t get_lightsource_type_end_index(std::vector<avk::lightsource>& aLightsources, avk::lightsource_type aLightsourceType)
	{
		auto it = std::upper_bound(std::begin(aLightsources), std::end(aLightsources), aLightsourceType, [](const avk::lightsource_type& a, const avk::lightsource& b){
			typedef std::underlying_type<avk::lightsource_type>::type EnumType;
			return static_cast<EnumType>(a) < static_cast<EnumType>(b.mType);
		});
		return static_cast<uint32_t>(std::distance(std::begin(aLightsources), it));
	}

	static uint32_t get_lightsource_type_end_index(avk::lightsource_type aLightsourceType)
	{
		return get_lightsource_type_end_index(get_lights(), aLightsourceType);
	}

	// create and initialize a lightsource editor
	static lights_editor create_lightsource_editor(avk::queue& aQueueToSubmitTo, bool aGuiEnabled)
	{
		auto lightsEd = lights_editor(aQueueToSubmitTo);
		lightsEd.set_gui_enabled(aGuiEnabled);
		lightsEd.add(helpers::get_lights());
		return lightsEd;
	}

	static bool is_lightsource_editor_visible() {
		auto lightsEd = avk::current_composition()->element_by_type<lights_editor>();
		if (lightsEd) {
			return lightsEd->is_gui_enabled();
		}
		else {
			return false;
		}
	}

	static void set_lightsource_editor_visible(bool aVisible)
	{
		auto lightsEd = avk::current_composition()->element_by_type<lights_editor>();
		if (lightsEd) {
			lightsEd->set_gui_enabled(aVisible);
		}
	}

	// get a vector of the active light sources
	// - if there is an instance of lights_editor around, take it from there
	// - otherwise just return our standard lightsources
	static std::vector<avk::lightsource> get_active_lightsources(int aLimitNumberOfPointLights = -1)
	{
		auto lightsEd = avk::current_composition()->element_by_type<lights_editor>();
		if (lightsEd) {
			return lightsEd->get_active_lights(aLimitNumberOfPointLights);
		}
		else {
			return get_lights();
		}
	}

	static bool are_lightsource_gizmos_enabled()
	{
		auto lightsEd = avk::current_composition()->element_by_type<lights_editor>();
		if (lightsEd) {
			return lightsEd->is_render_enabled();
		}
		else {
			return false;
		}
	}

	static void set_lightsource_gizmos_enabled(bool aEnabled)
	{
		auto lightsEd = avk::current_composition()->element_by_type<lights_editor>();
		if (lightsEd) {
			return lightsEd->set_render_enabled(aEnabled);
		}
	}

	// create and initialize camera presets
	static camera_presets create_camera_presets(avk::queue& aQueueToSubmitTo, bool aGuiEnabled)
	{
		auto camPresets = camera_presets(aQueueToSubmitTo);
		camPresets.set_gui_enabled(aGuiEnabled);

		ImVec2 window_pos  = ImVec2(207, 1);
		ImVec2 window_size = ImVec2(252, 139);
		// add preset for assignment 1
		camPresets.add_circular_motion("A1 autocam", { 0,1.5,0 }, { 2,2 }, { 0,1,0 }, glm::radians(45.f));

		// configure the gui layout
		camPresets.configure_gui(true, true, true, window_pos, window_size);

		return camPresets;
	}

	static bool is_camera_presets_editor_visible() {
		auto camPresets = avk::current_composition()->element_by_type<camera_presets>();
		if (camPresets) {
			return camPresets->is_gui_enabled();
		}
		else {
			return false;
		}
	}

	static void set_camera_presets_editor_visible(bool aVisible)
	{
		auto camPresets = avk::current_composition()->element_by_type<camera_presets>();
		if (camPresets) {
			camPresets->set_gui_enabled(aVisible);
		}
	}

	static std::unordered_map<std::string, std::tuple<vk::UniqueQueryPool, std::array<uint32_t, 2>, float>> sIntervals;

	static vk::QueryPool& add_timing_interval_and_get_query_pool(const std::string& aName)
	{
		auto iter = sIntervals.find(aName);
		if (iter == sIntervals.end()) {
			vk::QueryPoolCreateInfo queryPoolCreateInfo;
			queryPoolCreateInfo.setQueryCount(2);
			queryPoolCreateInfo.setQueryType(vk::QueryType::eTimestamp);

			iter = sIntervals.try_emplace(aName, avk::context().device().createQueryPoolUnique(queryPoolCreateInfo), std::array<uint32_t, 2>(), 0.0f).first;
		}
		return *std::get<0>(iter->second);
	}

	static void record_timing_interval_start(const vk::CommandBuffer& aCommandBuffer, const std::string& aName)
	{
		auto& queryPool = add_timing_interval_and_get_query_pool(aName);
		aCommandBuffer.resetQueryPool(queryPool, 0u, 2u);
		aCommandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eAllGraphics, queryPool, 0u);
	}

	static void record_timing_interval_end(const vk::CommandBuffer& aCommandBuffer, const std::string& aName)
	{
		auto& queryPool = add_timing_interval_and_get_query_pool(aName);
		aCommandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eAllGraphics, queryPool, 1u);
	}

	// request last timing interval from GPU and return averaged interval from previous measurements (in ms)
	static float get_timing_interval_in_ms(const std::string& aName)
	{
		auto iter = sIntervals.find(aName);
		if (iter == sIntervals.end()) {
			return 0.0f;
		}
		auto& [queryPool, timestamps, avgRendertime] = iter->second;
		avk::context().device().getQueryPoolResults(*queryPool, 0u, 2u, sizeof(timestamps), timestamps.data(), sizeof(uint32_t), vk::QueryResultFlagBits::eWait);
		float delta = (timestamps[1] - timestamps[0]) * avk::context().physical_device().getProperties().limits.timestampPeriod / 1000000.0f;
		avgRendertime = avgRendertime * 0.9f + delta * 0.1f;
		return avgRendertime;
	}

	static void clean_up_timing_resources()
	{
		sIntervals.clear();
	}
}

