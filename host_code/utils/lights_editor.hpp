#pragma once

#include <auto_vk_toolkit.hpp>
#include <imgui.h>
#include <glm/gtx/quaternion.hpp>

#include "imgui_manager.hpp"
#include "invokee.hpp"
#include "lightsource.hpp"
#include "math_utils.hpp"
#include "quake_camera.hpp"
#include "simple_geometry.hpp"
#include "vk_convenience_functions.hpp"

// TODO: use render_gizmos() instead of render() - this is currently complicated by ImGui !
// 
// TODO: use a generic renderpass? or pass in a renderpass as argument ?
// TODO: call render_gizmos() from main app or automatically draw them in render() ?
// TODO: safer handling of lightsources? (we just store pointers at the moment)
// TODO: initial size and location of gui window

class lights_editor : public avk::invokee
{
public:
	lights_editor(avk::queue& aQueueToSubmitTo, std::string aName = "lights_editor", bool aIsEnabled = true)
		: invokee(std::move(aName), aIsEnabled)
		, mQueue{ &aQueueToSubmitTo }
		, mCone{ &aQueueToSubmitTo }
		, mSphere{ &aQueueToSubmitTo }
	{}

	int execution_order() const override { return 1000; }

	void initialize() override
	{
		// NOTE: typically the main app's initialize() is called *before* this
		//printf("lights_editor: init\n");

		init_gizmos();
		init_gui();
	}

	void init_gizmos()
	{
		using namespace avk;

		// create pipeline
		mPipelineGizmos = context().create_graphics_pipeline_for(
			vertex_shader  ("shaders/utils/translucent_gizmo.vert.spv"),
			fragment_shader("shaders/utils/translucent_gizmo.frag.spv"),

			from_buffer_binding(0)->stream_per_vertex<glm::vec3>()->to_location(0), // aVertexPosition

			cfg::front_face::define_front_faces_to_be_counter_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(avk::context().main_window()->backbuffer_reference_at_index(0)),
			cfg::color_blending_config::enable_alpha_blending_for_attachment(0),
			cfg::depth_write::disabled(),

			context().create_renderpass({
					attachment::declare(avk::format_from_window_color_buffer(avk::context().main_window()), on_load::load, usage::color(0),      on_store::store),
					attachment::declare(avk::format_from_window_depth_buffer(avk::context().main_window()), on_load::load, usage::depth_stencil, on_store::store)
				}, {
					subpass_dependency(
						// Wait for previous write access to color and depth before rendering gizmos:
						subpass::external >> subpass::index(0),
						stage::late_fragment_tests             | stage::color_attachment_output  >>  stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
						access::depth_stencil_attachment_write | access::color_attachment_write  >>  access::depth_stencil_attachment_read                    | access::color_attachment_write
					),
					subpass_dependency(
						subpass::index(0) >> subpass::external,
						stage::color_attachment_output >> stage::none, // Semaphore afterwards, maybe? But it's probably going to be imgui_manager => should still be fine, because imgui_manager has subpass_dependencies too.
						access::color_attachment_write >> access::none
					)
				}
			),

			push_constant_binding_data{ shader_type::vertex | shader_type::fragment, 0, sizeof(PushConstantsGizmos) }
		);

		// create an updater, and add the pipeline (needed for window resize)
		mUpdater.emplace();
		mUpdater->on(avk::swapchain_changed_event(avk::context().main_window())).update(mPipelineGizmos);

		// create a command pool
		mCommandPool = avk::context().create_command_pool(mQueue->family_index(), vk::CommandPoolCreateFlagBits::eTransient);

		// create sphere and cone geometry
		mSphere.create_sphere();
		mCone.create_cone();

		mGizmosInited = true;
	}

	void init_gui()
	{
		auto imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
		if (!imguiManager) return;

		imguiManager->add_callback([this]() {
			using namespace ImGui;

			const float dragSpeedPos = 0.001f;
			const float dragSpeedDir = 0.001f;
			const float dragSpeedAtt = 0.01f;
			const float dragSpeedAng = 0.1f;
			const float dragSpeedFal = 0.01f;

			auto HelpMarker = [](const char* desc, bool sameLine = true) {
				if (sameLine) SameLine();
				TextDisabled("(?)");
				if (IsItemHovered()) {
					BeginTooltip();
					PushTextWrapPos(GetFontSize() * 35.0f);
					TextUnformatted(desc);
					PopTextWrapPos();
					EndTooltip();
				}
			};

			if (!mGuiEnabled) return;

			Begin("Lights", &mGuiEnabled);
			SetWindowPos(mInitialPosition, ImGuiCond_FirstUseEver);
			if (mInitialSize.x > 0.f) SetWindowSize(mInitialSize, ImGuiCond_FirstUseEver);

			if (mIdxPnt.size() > 1) {
				// *all* pointlights
				if (CollapsingHeader("ALL point lights")) {
					if (Button("Enable all"))  { for (auto idx : mIdxPnt) mLightEnabled[idx] = true;  }; SameLine();
					if (Button("Disable all")) { for (auto idx : mIdxPnt) mLightEnabled[idx] = false; }
					if (Button("Reset to initial state")) {
						for (auto idx : mIdxPnt) {
							*mLightsPtr[idx] = mLightsOriginal[idx];
						}
					}

					// TODO: do we need a uniform position offset for the point lights?
					auto p0 = mLightsPtr[mIdxPnt[0]];
					glm::vec3 atten = glm::vec3(p0->mAttenuationConstant, p0->mAttenuationLinear, p0->mAttenuationQuadratic);
					if (ColorEdit3 ("color", &p0->mColor.x, ImGuiColorEditFlags_NoInputs)) { for (auto idx : mIdxPnt) mLightsPtr[idx]->mColor = p0->mColor; }
					if (DragFloat3("atten", &atten.x, dragSpeedAtt)                      ) { for (auto idx : mIdxPnt) mLightsPtr[idx]->set_attenuation(glm::max(0.0f, atten.x), glm::max(0.0f, atten.y), glm::max(0.0f, atten.z)); }
					HelpMarker("Attenuation:\nconstant, linear, quadratic");
				}
			}

			std::vector<int> * indices[5] = { &mIdxAmb, &mIdxDir, &mIdxSpt, &mIdxPnt, &mIdxOth };
			const char * textSingle[5] = { "Ambient light",  "Directional light",  "Spot light",  "Point light",  "Other light"  };
			const char * textMulti [5] = { "Ambient lights", "Directional lights", "Spot lights", "Point lights", "Other lights" };
			int imgui_id = 0;
			for (int pass = 0; pass < 5; ++pass) {
				if (indices[pass]->size() > 0) {
					bool multiple = indices[pass]->size() > 1;
					int cnt = 0;
					if (CollapsingHeader(multiple ? textMulti[pass] : textSingle[pass])) {
						for (auto idx : *indices[pass]) {
							avk::lightsource * light = mLightsPtr[idx];
							bool ena = mLightEnabled[idx];

							PushID(imgui_id++);
							if (multiple) { Text("#%d:", cnt); SameLine(); }
							if (Checkbox("enabled", &ena)) mLightEnabled[idx] = ena;
							SameLine();
							ColorEdit3 ("color", &light->mColor.x, ImGuiColorEditFlags_NoInputs);
							SameLine();
							if (Button("reset")) {
								*mLightsPtr[idx] = mLightsOriginal[idx];
							}

							PushItemWidth(160);
							if (pass == 2 || pass == 3) { // spot, point
								DragFloat3("pos",   &light->mPosition.x, dragSpeedPos);
							}
							if (pass == 1 || pass == 2) { // dir, spot
								DragFloat3("direction", &light->mDirection.x, dragSpeedDir);
							}
							if (pass == 2) { // spot
								float angO = glm::degrees(light->mAngleOuterCone);
								float angI = glm::degrees(light->mAngleInnerCone);
								bool draggedO = false, draggedI = false;
								if (DragFloat("outer angle", &angO, dragSpeedAng, 0.0f, 359.9f, "%.1f")) { draggedO = true; light->mAngleOuterCone = glm::radians(angO); }
								if (DragFloat("inner angle", &angI, dragSpeedAng, 0.0f, 359.9f, "%.1f")) { draggedI = true; light->mAngleInnerCone = glm::radians(angI); }
								if (DragFloat("falloff", &light->mFalloff, dragSpeedFal) && light->mFalloff < 0.0f) light->mFalloff = 0.0f;
								if (draggedO && light->mAngleOuterCone < light->mAngleInnerCone) light->mAngleInnerCone = light->mAngleOuterCone;
								if (draggedI && light->mAngleOuterCone < light->mAngleInnerCone) light->mAngleOuterCone = light->mAngleInnerCone;
							}
							if (pass == 2 || pass == 3) { // spot, point
								glm::vec3 atten = glm::vec3(light->mAttenuationConstant, light->mAttenuationLinear, light->mAttenuationQuadratic);
								if (DragFloat3("atten", &atten.x, dragSpeedAtt)) light->set_attenuation(glm::max(0.0f, atten.x), glm::max(0.0f, atten.y), glm::max(0.0f, atten.z));
								HelpMarker("Attenuation:\nconstant, linear, quadratic");
							}
							PopItemWidth();

							PopID();
							cnt++;
							if (multiple) Separator();
						}
					}
				}
			}

			if (mGizmosInited) {
				if (CollapsingHeader("Gizmo settings")) {
					SliderFloat("Opacity",  &mGizmoParams.opacity, 0.01f, 1.0f);
					Text("Scale / attenuation contribution:");
					PushItemWidth(84);
					SliderFloat("##PL Scale",      &mGizmoParams.scalePL,        0.01f, 100.0f); SameLine();
					DragFloat  ("Point##PL Param", &mGizmoParams.paramPL, 0.01f, 0.01f, 10.0f);
					SliderFloat("##SL Scale",      &mGizmoParams.scaleSL,        0.01f,  4.0f);  SameLine();
					DragFloat  ("Spot##SL Param",  &mGizmoParams.paramSL, 0.01f, 0.01f, 10.0f);
					PopItemWidth();
					if (Button("Reset to defaults")) mGizmoParams = mGizmoParamsDefault;
				}
			}

			//static bool showdemo = false;
			//Checkbox("ImGui demo", &showdemo);

			End();

			//if (showdemo) ShowDemoWindow(&showdemo);
			});
	}

	/**	Configure the GUI
	*	@param	aInitialPos               Initial position of the GUI window (first-time use only)
	*	@param	aInitialSize              Initial size of the GUI window     (first-time use only)
	*/
	void configure_gui(std::optional<ImVec2> aInitialPos = std::nullopt, std::optional<ImVec2> aInitialSize = std::nullopt)
	{
		if (aInitialPos.has_value())  mInitialPosition = aInitialPos.value();
		if (aInitialSize.has_value()) mInitialSize     = aInitialSize.value();
	}

	// add a light source
	void add(avk::lightsource * ptrLightsource)
	{
		int index = static_cast<int>(mLightsPtr.size());
		mLightsPtr.push_back(ptrLightsource);
		avk::lightsource copy = *ptrLightsource;
		mLightsOriginal.push_back(copy);
		mLightEnabled.push_back(true);

		switch(ptrLightsource->mType) {
		case avk::lightsource_type::ambient:		mIdxAmb.push_back(index);	break;
		case avk::lightsource_type::directional:	mIdxDir.push_back(index);	break;
		case avk::lightsource_type::point:			mIdxPnt.push_back(index);	break;
		case avk::lightsource_type::spot:			mIdxSpt.push_back(index);	break;
		default:
			LOG_WARNING("Light source type not supported in lights editor");
			mIdxOth.push_back(index);
			break;
		}
	}

	// add a whole vector of light sources
	void add(std::vector<avk::lightsource> &vecLightsource)
	{
		for (auto &p : vecLightsource) add(&p);
	}

	std::vector<avk::lightsource> get_active_lights(int aLimitNumberOfPointLights = -1)
	{
		std::vector<avk::lightsource> result;
		int countPL = 0;
		for (auto i = 0; i < mLightsPtr.size(); ++i) {
			if (mLightEnabled[i]) {
				if (aLimitNumberOfPointLights >= 0 && mLightsPtr[i]->mType == avk::lightsource_type::point) {
					countPL++;
					if (countPL > aLimitNumberOfPointLights) continue;
				}
				result.push_back(*mLightsPtr[i]);
			}
		}
		return result;
	}

	bool is_gui_enabled() { return mGuiEnabled; }
	void set_gui_enabled(bool aEnabled) { mGuiEnabled = aEnabled; }

	// TODO: make this render_gizmos() once problems with ImGui are solved
	void render() override {
		// get the camera
		auto cam = avk::current_composition()->element_by_type<avk::quake_camera>();
		if (nullptr == cam) {
			return;
		}

		// record command buffer
		auto cmdBfr = mCommandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		cmdBfr->begin_recording();
		cmdBfr->record(avk::command::begin_render_pass_for_framebuffer(mPipelineGizmos->renderpass_reference(), avk::context().main_window()->current_backbuffer_reference()));
		draw_gizmos(cmdBfr, cam->projection_and_view_matrix());
		cmdBfr->record(avk::command::end_render_pass());
		cmdBfr->end_recording();

		// SUBMIT, and establish necessary sync:
		auto mainWnd = avk::context().main_window(); 
		auto submission = mQueue->submit(cmdBfr.as_reference());

		// If this is the first render call, then consume the image available semaphore:
		if (!mainWnd->has_consumed_current_image_available_semaphore()) {
			submission
				.waiting_for(mainWnd->consume_current_image_available_semaphore() >> avk::stage::early_fragment_tests);
		}
		// IF we assume that imgui_manager ALWAYS runs afterwards, we can spare ourselves another window::add_present_dependency_for_current_frame

		submission.submit();
		avk::context().main_window()->handle_lifetime(std::move(cmdBfr));
	}

private:
	void draw_gizmos(avk::command_buffer & cmd, const glm::mat4 & projectionViewMatrix)
	{
		// must already have started a renderpass

		if (!mGizmosInited) return;

		cmd->record(avk::command::bind_pipeline(mPipelineGizmos.as_reference()));
		for (auto idx : mIdxPnt) {
			if (mLightEnabled[idx]) {
				avk::lightsource *p = mLightsPtr[idx];

				auto d = mGizmoParams.paramPL;
				auto s = mGizmoParams.scalePL / (p->mAttenuationConstant + p->mAttenuationLinear * d + p->mAttenuationQuadratic * d * d);

				PushConstantsGizmos pushConstants;
				pushConstants.pvmtMatrix = projectionViewMatrix * glm::translate(p->mPosition) * glm::scale(glm::vec3(s));
				pushConstants.uColor = glm::vec4(p->mColor, mGizmoParams.opacity);
				cmd->handle().pushConstants(mPipelineGizmos->layout_handle(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pushConstants), &pushConstants);
				cmd->record(avk::command::draw_indexed(mSphere.mIndexBuffer.as_reference(), mSphere.mPositionsBuffer.as_reference()));
			}
		}
		for (auto idx : mIdxSpt) {
			if (mLightEnabled[idx]) {
				avk::lightsource *p = mLightsPtr[idx];

				auto d = mGizmoParams.paramSL;
				auto s = mGizmoParams.scaleSL / (p->mAttenuationConstant + p->mAttenuationLinear * d + p->mAttenuationQuadratic * d * d);
				auto angleScale = tan(p->mAngleOuterCone * 0.5f);

				PushConstantsGizmos pushConstants;
				pushConstants.pvmtMatrix = projectionViewMatrix
					* glm::translate(p->mPosition)
					* glm::toMat4(avk::rotation_between_vectors(glm::vec3(0,1,0), p->mDirection))
					* glm::scale(glm::vec3(s * angleScale, s, s * angleScale));
				pushConstants.uColor = glm::vec4(p->mColor, mGizmoParams.opacity);
				cmd->handle().pushConstants(mPipelineGizmos->layout_handle(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pushConstants), &pushConstants);
				cmd->record(avk::command::draw_indexed(mCone.mIndexBuffer.as_reference(), mCone.mPositionsBuffer.as_reference()));
			}
		}
	}

	avk::queue* mQueue;
	avk::command_pool mCommandPool;

	std::vector<avk::lightsource *> mLightsPtr;
	std::vector<avk::lightsource> mLightsOriginal;
	std::vector<int> mIdxAmb, mIdxDir, mIdxPnt, mIdxSpt, mIdxOth;
	std::vector<bool> mLightEnabled;

	struct PushConstantsGizmos {
		glm::mat4 pvmtMatrix;
		glm::vec4 uColor;
	};

	avk::graphics_pipeline mPipelineGizmos;
	simple_geometry mSphere, mCone;

	struct {
		float opacity = 0.3f;
		float scalePL = 8.0f;
		float paramPL = 3.0f;
		float scaleSL = 0.7f;
		float paramSL = 1.5f;
	} mGizmoParamsDefault, mGizmoParams;

	bool mGizmosInited = false;
	bool mGuiEnabled = true;

	ImVec2 mInitialPosition = { 1.0f, 283.0f };
	ImVec2 mInitialSize     = { 262.0f, 287.0f };
};


