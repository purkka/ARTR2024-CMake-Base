#pragma once

#include <auto_vk_toolkit.hpp>
#include <imgui.h>
#include <glm/gtx/quaternion.hpp>

#include "bezier_curve.hpp"
#include "catmull_rom_spline.hpp"
#include "cp_interpolation.hpp"
#include "cubic_uniform_b_spline.hpp"
#include "orbit_camera.hpp"
#include "quadratic_uniform_b_spline.hpp"

// TODO! light gizmos and path rendering will probably break if main uses a different renderpass setup!

// TODO: paths: constant speed, closed paths
// TODO: paths: autoclose ?

class camera_presets : public avk::invokee
{
public:
	enum class focus_type
	{
		towards_point   = 0,
		away_from_point = 1,
		forward         = 2,
		backward        = 3,
		none            = 4 // <- keep this the last
	};

	enum class path_type
	{
		bezier            = 0,
		quadratic_bspline = 1,
		cubic_bspline     = 2,
		catmull_rom       = 3
	};

	camera_presets(avk::queue& aQueueToSubmitTo, std::string aName = "camera_presets", bool aIsEnabled = true)
		: invokee(std::move(aName), aIsEnabled)
		, mQueue{ &aQueueToSubmitTo }
	{}

	int execution_order() const override { return 100; }

	/**	Configure the GUI
	*	@param	aAllowToAddPresets        Show controls to add new presets             [default = true]
	*	@param	aShowCameraInfo           Show camera location and direction           [default = true]
	*	@param	aAllowPathVisualization   Allow to visualize camera paths when editing [default = true]
	*	@param	aInitialPos               Initial position of the GUI window (first-time use only)
	*	@param	aInitialSize              Initial size of the GUI window     (first-time use only)
	*/
	void configure_gui(bool aAllowToAddPresets, bool aShowCameraInfo, bool aAllowPathVisualization, std::optional<ImVec2> aInitialPos = std::nullopt, std::optional<ImVec2> aInitialSize = std::nullopt)
	{
		mAllowAddPresets    = aAllowToAddPresets;
		mShowCameraInfo     = aShowCameraInfo;
		mAllowVisualizePath = aAllowPathVisualization;
		if (aInitialPos.has_value())  mInitialPosition = aInitialPos.value();
		if (aInitialSize.has_value()) mInitialSize     = aInitialSize.value();
	}

	/** Add a location preset
	*	@param	aName           Name of the preset
	*	@param	aPosition       Camera position
	*	@param	aRotation       Camera rotation
	*/
	std::string add_location(const std::string aName, const glm::vec3 &aPosition, const glm::quat &aRotation)
	{
		preset_data p;
		p.type = preset_type::location;
		p.name = unique_name(aName);
		p.translation = aPosition;
		p.rotation = aRotation;
		p.direction = glm::normalize(aRotation * glm::vec3(0, 0, -1));
		mPresets.push_back(p);
		return p.name;
	}

	/** Add a location preset
	*	@param	aName           Name of the preset
	*	@param	aPosition       Camera position
	*	@param	aDirection      Camera view direction
	*/
	std::string add_location(const std::string aName, const glm::vec3 &aPosition, const glm::vec3 &aDirection)
	{
		return add_location(aName, aPosition, camera_rotation_from_direction(aDirection));
	}

	/** Change a location preset
	*	@param	aName             Name of the preset
	*	@param	aPosition         Camera position
	*	@param	aRotation         Camera rotation
	*	@param	aCreateIfNotFound Create the preset if it does not exist yet
	*/
	void change_location(const std::string aName, const glm::vec3 &aPosition, const glm::quat &aRotation, bool aCreateIfNotFound = true) {
		auto p = find_preset(aName);
		if (p) {
			p->translation = aPosition;
			p->rotation = aRotation;
			p->direction = glm::normalize(aRotation * glm::vec3(0, 0, -1));
		}
		else if (aCreateIfNotFound) {
			add_location(aName, aPosition, aRotation);
		}
	}

	/** Change a location preset
	*	@param	aName             Name of the preset
	*	@param	aPosition         Camera position
	*	@param	aDirection        Camera view direction
	*	@param	aCreateIfNotFound Create the preset if it does not exist yet
	*/
	void change_location(const std::string aName, const glm::vec3 &aPosition, const glm::vec3 &aDirection, bool aCreateIfNotFound = true) {
		change_location(aName, aPosition, camera_rotation_from_direction(aDirection), aCreateIfNotFound);
	}

	/** Add a circular/elliptical motion preset
	*	@param	aName           Name of the preset
	*	@param	aCenter         Center of ellipse
	*	@param	aRadiusXZ       Radius of ellipse (in x and z-direction)
	*	@param	aFocus          Focus point to keep looking at (see aFocusType)
	*	@param	aAngularSpeed   Movement speed in radians per second
	*	@param	aStartAngle     Start angle of motion in radians
	*   @param  aFocusType      How to set the camera view direction when the motion is active:
	*                             towards_point  :  look towards focus point
	*                             away_from_point:  look away from focus point
	*                             forward        :  look forward in the motion direction
	*                             backward       :  look backwards (against the motion direction)
	*                             none           :  don't affect the camera view direction at all (a.k.a. free-look)
	*
	*  The motion runs at constant angular speed, starting with angle 0 on the positive x-axis and rotates counter-clockwise (when looked at from +y) in the x-z plane.
	*  The rotation direction can be inverted by specifying a negative aAngularSpeed.
	*/
	std::string add_circular_motion(const std::string aName, const glm::vec3 &aCenter, const glm::vec2 &aRadiusXZ, const glm::vec3 &aFocus, const float aAngularSpeed, float aStartAngle = 0.0f, focus_type aFocusType = focus_type::towards_point)
	{
		preset_data p;
		p.type = preset_type::circular;
		p.name = unique_name(aName);
		p.center = aCenter;
		p.focus = aFocus;
		p.radius_xz = aRadiusXZ;
		p.angular_speed = aAngularSpeed;
		p.start_angle = aStartAngle;
		p.focus_type = aFocusType;
		mPresets.push_back(p);
		return p.name;
	}

	/** Add an interpolated camera path motion preset
	*	@param	aName           Name of the preset
	*   @param  aType           Type of the interpolation curve (bezier, quadratic_bspline, cubic_bspline or catmull_rom)
	*	@param	aDuration       Time it takes to traverse the full path (in seconds)
	*   @param  aCyclic         Restart the motion when the end point has been reached
	*   @param  aControlPoints  Control points of the path
	*   @param  aFocusType      How to set the camera view direction when the motion is active ( see add_circular_motion() )
	*	@param	aFocus          Focus point to keep looking at                                 ( see add_circular_motion() )
	* 
	*   Note that Catmull-Rom interpolation requires two extra control points (lead-in and lead-out).
	*/
	std::string add_path(const std::string aName, path_type aType, float aDuration, bool aCyclic, const std::vector<glm::vec3> &aControlPoints, focus_type aFocusType = focus_type::forward, const glm::vec3 aFocus = {0,0,0})
	{
		preset_data p;
		p.type = preset_type::path;
		p.name = unique_name(aName);
		p.path_type = aType;
		p.path_duration = aDuration;
		p.path_cyclic = aCyclic;
		p.path_control_points = aControlPoints;
		p.path_interpolation()->set_control_points(p.path_control_points);
		p.focus = aFocus;
		p.focus_type = aFocusType;
		mPresets.push_back(p);
		return p.name;
	}

	/** Invoke a preset
	*	@param	aName           Name of the preset
	*
	*   If it is a location preset, the camera is moved and oriented as specified.
	*   If it is a motion preset (circular or path), the motion is (re)started.
	*/
	void invoke_preset(const std::string aName)
	{
		invoke_preset(find_preset(aName));
	}

	/** Stop any active motion preset
	*/
	void stop_all_motion()
	{
		stop_all_motion(nullptr);
	}

	/** Stop a specific motion preset
	*	@param	aName           Name of the preset
	*/
	void stop_preset(const std::string aName)
	{
		auto p = find_preset(aName);
		if (p) p->motion_active = false;
	}

	/** Test if a (motion) preset is active
	*	@param	aName           Name of the preset
	*/
	bool is_preset_active(const std::string aName)
	{
		auto p = find_preset(aName);
		return (p && p->motion_active);
	}

	/** Lock or unlock a preset (prevents deletion and editing via GUI)
	*	@param	aName           Name of the preset
	*   @param  aLocked         true to lock, false to unlock
	*/
	void lock_preset(const std::string aName, bool aLocked = true)
	{
		auto p = find_preset(aName);
		if (p) p->locked = aLocked;
	}

	/** Lock (or unlock) all currently stored presets
	*   @param  aLocked         true to lock, false to unlock
	*/
	void lock_all_presets(bool aLocked = true)
	{
		for (auto &p : mPresets) p.locked = aLocked;
	}

	bool is_gui_enabled() { return mGuiEnabled; }
	void set_gui_enabled(bool aEnabled) { mGuiEnabled = aEnabled; }

	void initialize() override
	{
		init_gui();
	}

	void update() override
	{
		// check if any motion preset is active
		auto quakeCam = avk::current_composition()->element_by_type<avk::quake_camera>();
		auto orbitCam = avk::current_composition()->element_by_type<avk::orbit_camera>();
		if (!quakeCam || !orbitCam) return;
		for (auto &p : mPresets) {
			if (p.motion_active) {
				float time = static_cast<float>(avk::context().get_time());
				if (p.type == preset_type::circular) {
					float angle = fmod((time - p.motion_start_time) * p.angular_speed + p.start_angle, glm::two_pi<float>());
					glm::vec3 pos = p.center + glm::vec3(cos(angle) * p.radius_xz[0], 0, sin(angle) * p.radius_xz[1]);
					quakeCam->set_translation(pos);
					orbitCam->set_translation(pos);
					glm::vec3 dir;
					switch (p.focus_type) {
					case focus_type::towards_point:		dir = p.focus - pos;	break;
					case focus_type::away_from_point:	dir = pos - p.focus;	break;
					case focus_type::forward:
					case focus_type::backward:
					{
						// TODO calc proper tangent to the ellipse
						const float da = glm::radians(5.0f) * glm::sign(p.angular_speed);
						glm::vec3 pos2 = p.center + glm::vec3(cos(angle + da) * p.radius_xz[0], 0, sin(angle + da) * p.radius_xz[1]);
						dir = pos2 - pos;
						if (p.focus_type == focus_type::backward) dir = -dir;
						break;
					}
					default:							dir = glm::vec3(0);		break;
					}
					if (glm::length2(dir) > 0) {
						quakeCam->look_along(dir);
						orbitCam->look_along(dir);
					}
				} else if (p.type == preset_type::path) {
					float tSpline = 0.0f;

					if (!p.is_path_valid()) p.motion_active = false;

					if (p.motion_active) {
						tSpline = (time - p.motion_start_time) / p.path_duration;
						if (tSpline > 1.0f) {
							if (p.path_cyclic) {
								tSpline = glm::fract(tSpline);
							} else {
								p.motion_active = false;
							}
						}
					}
					if (p.motion_active) {
						glm::vec3 pos = p.path_interpolation()->value_at(tSpline);
						//glm::vec3 dir = p.path_interpolation()->slope_at(tSpline);
						glm::vec3 dir;
						switch (p.focus_type) {
						case focus_type::towards_point:		dir = p.focus - pos;	break;
						case focus_type::away_from_point:	dir = pos - p.focus;	break;
						case focus_type::forward:			dir =  p.path_interpolation()->slope_at(tSpline);	break;
						case focus_type::backward:			dir = -p.path_interpolation()->slope_at(tSpline);	break;
						default:							dir = glm::vec3(0);		break;
						}

						quakeCam->set_translation(pos);
						orbitCam->set_translation(pos);
						if (glm::length2(dir) > 0) {
							quakeCam->look_along(dir);
							orbitCam->look_along(dir);
						}
					}

				}
			}
		}
	}

	void render() override
	{
		if (!mVisualizePath) return;
		if (!mRenderingInited) init_rendering();

		auto preset = find_preset(mVisualizePathPresetName);
		if (!preset) return;
		if (!preset->is_path_valid()) return;
		if (preset->path_control_points.size() > MAX_CONTROL_POINTS_TO_VISUALIZE) return;

		auto cam = avk::current_composition()->element_by_type<avk::quake_camera>();
		if (!cam) return;

		auto fif = avk::context().main_window()->in_flight_index_for_frame();

		// fill path and control points vertex buffer
		std::vector<glm::vec3> pathPoints(MAX_POINTS_TO_VISUALIZE);
		for (int i = 0; i < MAX_POINTS_TO_VISUALIZE; ++i) {
			float t = static_cast<float>(i) / (MAX_POINTS_TO_VISUALIZE - 1);
			pathPoints[i] = preset->path_interpolation()->value_at(t);
		}
		auto &ctrlPoints = preset->path_control_points;
		auto fence = avk::context().record_and_submit_with_fence({
			mVertexBufferVisPath1[fif]->fill(pathPoints.data(), 0, 0, pathPoints.size() * sizeof(pathPoints[0])),
			mVertexBufferVisPath2[fif]->fill(ctrlPoints.data(), 0, 0, ctrlPoints.size() * sizeof(ctrlPoints[0]))
		}, *mQueue);
		fence->wait_until_signalled(); // A fence means a heavy barrier here, but ok here, as this is only used for designing paths // TODO: Use semaphore?!

		// record command buffer
		auto cmdBfr = mCommandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		cmdBfr->begin_recording();
		cmdBfr->record(avk::command::begin_render_pass_for_framebuffer(mPipelineVisPath1->renderpass_reference(), avk::context().main_window()->current_backbuffer_reference()));
		cmdBfr->record(avk::command::bind_pipeline(mPipelineVisPath1.as_reference()));
		PushConstantsVisPath pushConstants = {};
		pushConstants.mViewProjMatrix = cam->projection_and_view_matrix();
		pushConstants.mColor          = glm::vec4(1,0,0,1);
		pushConstants.mColor2         = glm::vec4(0);
		pushConstants.mVertexToHighlight = -1;

		cmdBfr->handle().pushConstants(mPipelineVisPath1->layout_handle(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pushConstants), &pushConstants);
		cmdBfr->record(avk::command::draw_vertices(static_cast<uint32_t>(pathPoints.size()), 1u, 0u, 0u, mVertexBufferVisPath1[fif].as_reference()));
		cmdBfr->record(avk::command::bind_pipeline(mPipelineVisPath2.as_reference()));
		pushConstants.mColor          = glm::vec4(0, 1, 0, 10);
		pushConstants.mColor2         = glm::vec4(0, 1, 1, 0);
		pushConstants.mVertexToHighlight = mVisualizePathCurrentPointIndex;
		cmdBfr->handle().pushConstants(mPipelineVisPath2->layout_handle(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pushConstants), &pushConstants);
		cmdBfr->record(avk::command::draw_vertices(static_cast<uint32_t>(ctrlPoints.size()), 1u, 0u, 0u, mVertexBufferVisPath2[fif].as_reference()));

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
	enum class preset_type {
		location = 0,
		path     = 1,
		circular = 2
	};

	struct preset_data {
		preset_type	type = preset_type::location;
		std::string name;
		// single location:
		glm::vec3 translation;
		glm::quat rotation;
		glm::vec3 direction; // only used in editor
							 // circular
		glm::vec3 center, focus;
		glm::vec2 radius_xz;
		float angular_speed, start_angle;
		// path
		path_type path_type;
		std::vector<glm::vec3> path_control_points;
		float path_duration;
		bool path_cyclic;
		// all motion
		focus_type focus_type;
		bool motion_active = false;
		float motion_start_time;

		bool locked = false;

		avk::cp_interpolation * path_interpolation() {
			switch (path_type) {
			case path_type::bezier:				return &bezier_curve;
			case path_type::quadratic_bspline:	return &quadratic_uniform_b_spline;
			case path_type::cubic_bspline:		return &cubic_uniform_b_spline;
			case path_type::catmull_rom:		return &catmull_rom_spline;
			default:							return &bezier_curve;
			}
		}
		bool is_path_valid() {
			bool result = (type == preset_type::path) && (path_control_points.size() > 2) && (path_duration > 0.0f);
			if (path_type == path_type::catmull_rom && path_control_points.size() < 4) result = false;
			return result;
		}
	private:
		avk::bezier_curve bezier_curve;
		avk::quadratic_uniform_b_spline quadratic_uniform_b_spline;
		avk::cubic_uniform_b_spline cubic_uniform_b_spline;
		avk::catmull_rom_spline catmull_rom_spline;
	};


	void init_gui()
	{
		auto imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
		if (!imguiManager) return;

		imguiManager->add_callback([this]() {
			using namespace ImGui;

			if (!mGuiEnabled) return;

			auto cam = avk::current_composition()->element_by_type<avk::quake_camera>();

			Begin("Camera Presets", &mGuiEnabled);
			SetWindowPos(mInitialPosition, ImGuiCond_FirstUseEver);
			if (mInitialSize.x > 0.f) SetWindowSize(mInitialSize, ImGuiCond_FirstUseEver);

			if (!cam) {
				Text("CAMERA NOT FOUND!");
			} else {

				const float dragSpeedPos = 0.001f;
				const float dragSpeedDir = 0.001f;
				const float dragSpeedAng = 0.1f;
				const char * focusTypeComboStrings = "to focus point\0away from focus point\0forward\0backward\0free look\0";

				static std::string presetToEdit = "";
				static std::string codeToShow = "";
				bool showCode = false;

				int iDelete = -1;
				bool startEditing = false;
				bool motionActive = false;
				PushID("Presets");
				for (int iPreset = 0; iPreset < static_cast<int>(mPresets.size()); ++iPreset) {
					PushID(iPreset);
					auto &p = mPresets[iPreset];
					motionActive = motionActive || p.motion_active;
					float spacing = 60.f;
					bool colored = p.motion_active;
					if (colored) {
						ImGui::PushStyleColor(ImGuiCol_Button,        (ImVec4)ImColor::HSV(3 / 7.0f, 0.6f, 0.6f));
						ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(3 / 7.0f, 0.7f, 0.7f));
						ImGui::PushStyleColor(ImGuiCol_ButtonActive,  (ImVec4)ImColor::HSV(3 / 7.0f, 0.8f, 0.8f));
						spacing += 44;
					}
					if (Button(p.name.c_str(), ImVec2(-spacing, 0))) invoke_preset(p.name);
					if (colored) {
						ImGui::PopStyleColor(3);
						SameLine();
						if (Button("Stop")) stop_all_motion();
					}
					if (!p.locked) {
						SameLine();
						if (Button("Ed##Edit preset")) {
							if (presetToEdit == p.name) {
								presetToEdit = "";
							} else {
								presetToEdit = p.name;
								startEditing = true;
							}
						}
						SameLine();
						if (Button("X##Delete preset")) iDelete = iPreset;
					}
					PopID();
				}
				PopID();
				if (iDelete >= 0) mPresets.erase(mPresets.begin() + iDelete);

				if (mAllowAddPresets) {
					Separator();
					if (Button("+Location")) add_location("", cam->translation(), cam->rotation());
					SameLine();
					if (Button("+Circle")) {
						presetToEdit = add_circular_motion("", { 0,0,0 }, { 1,1 }, { 0,0,0 }, glm::radians(45.0f));
						startEditing = true;
					}
					SameLine();
					if (Button("+Path")) {
						presetToEdit = add_path("", path_type::bezier, 10.0f, false, { {0,0,0}, {1,0,0}, {1,0,1}, {0,0,1} });
						startEditing = true;
					}
				}

				bool visualizedPathBefore = mVisualizePath;
				mVisualizePath = false;
				if (!presetToEdit.empty()) {
					auto p = find_preset(presetToEdit);
					if (!p) {
						presetToEdit = "";
					} else {
						Separator();
						PushID("EditPreset");
						Text("Edit preset \"%s\":", p->name.c_str());
						static char txtNewName[MAX_NAME_LEN + 1] = "";
						if (startEditing) {
							strncpy_s(txtNewName, p->name.c_str(), MAX_NAME_LEN);
						}
						PushItemWidth(140);
						InputText("##NewName", txtNewName, _ARRAYSIZE(txtNewName));
						SameLine();
						if (Button("Rename")) {
							p->name = unique_name(std::string(txtNewName), p);
							presetToEdit = p->name;
							strncpy_s(txtNewName, p->name.c_str(), MAX_NAME_LEN);
						}
						if (p->type == preset_type::location) {
							auto oldPos = p->translation;
							auto oldRot = p->rotation;
							DragFloat3("Position", &p->translation.x, dragSpeedPos);
							if (DragFloat3("Direction", &p->direction.x, dragSpeedDir)) p->rotation = camera_rotation_from_direction(p->direction);
							if (Button("Set from camera")) {
								p->translation = cam->translation();
								p->rotation = cam->rotation();
							}
							if (p->translation != oldPos || p->rotation != oldRot) invoke_preset(p);
						} else if (p->type == preset_type::circular) {
							float speedDeg = glm::degrees(p->angular_speed);
							float angleDeg = glm::degrees(p->start_angle);
							DragFloat3("Center", &p->center.x, dragSpeedPos);
							int ftype = static_cast<int>(p->focus_type);
							if (Combo("Look", &ftype, focusTypeComboStrings)) p->focus_type = static_cast<focus_type>(ftype);
							DragFloat3("Focus", &p->focus.x, dragSpeedPos);
							DragFloat2("Radius x/z", &p->radius_xz.x, dragSpeedPos);
							if (DragFloat("Deg/sec", &speedDeg, dragSpeedAng)) p->angular_speed = glm::radians(speedDeg);
							if (DragFloat("Start angle", &angleDeg, dragSpeedAng)) p->start_angle = glm::radians(angleDeg);
						} else if (p->type == preset_type::path) {
							int pathtype = static_cast<int>(p->path_type);
							if (Combo("Type", &pathtype, "Bezier Curve\0Quadratic B-Spline\0Cubic B-Spline\0Catmull-Rom Spline\0")) {
								p->path_interpolation()->set_control_points({}); // clear control points from old interpolator
								p->path_type = static_cast<path_type>(pathtype);
								p->path_interpolation()->set_control_points(p->path_control_points); // set control points in new interpolator
							}
							InputFloat("Duration (sec)", &p->path_duration);
							Checkbox("Cyclic", &p->path_cyclic);
							int ftype = static_cast<int>(p->focus_type);
							if (Combo("Look##Look_path", &ftype, focusTypeComboStrings)) p->focus_type = static_cast<focus_type>(ftype);
							DragFloat3("Focus##Focus_path", &p->focus.x, dragSpeedPos);

							bool changed = false;
							int delPos = -1;
							int addPos = -1;
							int moveUp = -1;
							int moveDn = -1;

							auto &points = p->path_control_points;

							Separator();
							Text("Control points:");
							//BeginChild("scrollbox", ImVec2(0, 200));
							PushID("PathControlPoints");
							for (int i = 0; i < static_cast<int>(points.size()); ++i) {
								PushID(i);
								if (p->path_type == path_type::catmull_rom && (i == 1 || i == static_cast<int>(points.size()) - 1)) Separator(); // lead in/out pointd

								if (visualizedPathBefore && (i == mVisualizePathCurrentPointIndex)) {
									TextColored(ImVec4(0,1,1,1), "#%02d", i);
								} else {
									Text("#%02d", i);
								}
								SameLine();

								PushItemWidth(160);
								if (DragFloat3("##pos", &(points[i].x), dragSpeedPos, 0.f, 0.f, "%.2f")) {
									changed = true;
									mVisualizePathCurrentPointIndex = i;
								}
								PopItemWidth();

								SameLine();
								if (Button("...##DoCtrlPointPopup")) {
									ImGui::OpenPopup("CtrlPointPopup");
									mVisualizePathCurrentPointIndex = i;
								}

								const char * ctrlPtMenuitems[] = { "Set to current pos", "Delete control point ", "Add new control point", "Move up", "Move down", "Jump to" };
								static int selectedCtrlPtMenuitem = -1;
								if (ImGui::BeginPopup("CtrlPointPopup"))
								{
									for (int i = 0; i < IM_ARRAYSIZE(ctrlPtMenuitems); i++)
										if (ImGui::Selectable(ctrlPtMenuitems[i]))
											selectedCtrlPtMenuitem = i;
									ImGui::EndPopup();
								}
								if (selectedCtrlPtMenuitem >= 0) {
									switch(selectedCtrlPtMenuitem) {
									case 0: points[i] = cam->translation(); changed = true; break;
									case 1: delPos = i; break;
									case 2: addPos = i; break;
									case 3: moveUp = i; break;
									case 4: moveDn = i; break;
									case 5: cam->set_translation(points[i]); break;
									}
									selectedCtrlPtMenuitem = -1;
								}

								//SameLine(); if (Button("set")) { points[i] = cam->translation(); changed = true; }
								//SameLine(); if (Button("X")) delPos = i;
								//SameLine(); if (Button("+")) addPos = i;
								//SameLine(); if (Button("^")) moveUp = i;
								//SameLine(); if (Button("v")) moveDn = i;
								//SameLine(); if (Button("go")) { cam->set_translation(points[i]); }
								PopID();
							}
							PopID();
							//EndChild();
							Separator();

							if (addPos >= 0)												{ points.insert(points.begin() + addPos + 1, cam->translation());	changed = true; }
							if (delPos >= 0)												{ points.erase (points.begin() + delPos);							changed = true; }
							if (moveUp >  0)												{ std::swap(points[moveUp - 1], points[moveUp]);					changed = true; }
							if (moveDn >= 0 && moveDn < static_cast<int>(points.size())-1)	{ std::swap(points[moveDn + 1], points[moveDn]);					changed = true; }
							if (changed) p->path_interpolation()->set_control_points(points);

						}

						if (p->type == preset_type::circular || p->type == preset_type::path) {
							if (p->motion_active) {
								if (Button("Stop")) stop_all_motion();
							} else {
								if (Button("Start")) invoke_preset(p);
							}
							if (mAllowVisualizePath && p->type == preset_type::path) {
								static bool vis = false;
								SameLine();
								Checkbox("Visualize", &vis);
								if (vis) {
									mVisualizePath = true;
									mVisualizePathPresetName = p->name;
								}
							}
						}

						PopItemWidth();
						if (Button("Close editor")) presetToEdit = "";
						SameLine();
						if (Button("Show code")) {
							codeToShow = generate_code(p);
							showCode = true;
						}
						PopID();
					}
				}

				if (motionActive) {
					Separator();
					if (Button("Stop current motion")) stop_all_motion();
				}

				if (mShowCameraInfo) {
					Separator();
					Text("Camera:");
					auto pos = cam->translation();
					auto rot = cam->rotation();
					auto dir = rot * glm::vec3(0, 0, -1);
					Text("Pos: %.2f %.2f %.2f", pos.x, pos.y, pos.z);
					Text("Dir: %.2f %.2f %.2f", dir.x, dir.y, dir.z);
					//Text("Rot: %.3f %.2f %.2f %.2f", rot[0], rot[1], rot[2], rot[3]);
				}

				// code window
				if (showCode) OpenPopup("code_window");

				ImVec2 center = ImGui::GetMainViewport()->GetCenter();
				ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
				if (BeginPopupModal("code_window" /* , nullptr, ImGuiWindowFlags_AlwaysAutoResize */)) {
					Text("Generated code:");
					InputTextMultiline("##source", (char *)codeToShow.c_str(), codeToShow.length() + 1, ImVec2(-FLT_MIN, -ImGui::GetTextLineHeight() * 2), ImGuiInputTextFlags_ReadOnly);
					//InputTextMultiline("##source", (char *)codeToShow.c_str(), codeToShow.length() + 1, ImVec2(400, ImGui::GetTextLineHeight() * 16), ImGuiInputTextFlags_ReadOnly);

					if (Button("Close")) CloseCurrentPopup();
					EndPopup();
				}
			}

			//static bool showdemo = false;
			//Checkbox("Show ImGui demo", &showdemo);

			End();

			//if (showdemo) ShowDemoWindow(&showdemo);
		});
	}

	void init_rendering() {
		if (mRenderingInited) return;

		using namespace avk;

		// create pipeline
		mPipelineVisPath1 = context().create_graphics_pipeline_for(
			vertex_shader("shaders/utils/campreset_vispath.vert.spv"),
			fragment_shader("shaders/utils/campreset_vispath.frag.spv"),

			from_buffer_binding(0)->stream_per_vertex<glm::vec3>()->to_location(0), // inPosition

			cfg::primitive_topology::line_strip,
			cfg::viewport_depth_scissors_config::from_framebuffer(avk::context().main_window()->backbuffer_reference_at_index(0)),
			cfg::depth_write::disabled(),
			//cfg::depth_test::disabled(),

			attachment::declare(format_from_window_color_buffer(context().main_window()), on_load::load, usage::color(0),      on_store::store),
			attachment::declare(format_from_window_depth_buffer(context().main_window()), on_load::load, usage::depth_stencil, on_store::store),

			push_constant_binding_data{ shader_type::vertex | shader_type::fragment, 0, sizeof(PushConstantsVisPath) }
		);

		mPipelineVisPath2 = context().create_graphics_pipeline_for(
			vertex_shader  ("shaders/utils/campreset_vispath.vert.spv"),
			fragment_shader("shaders/utils/campreset_vispath.frag.spv"),

			from_buffer_binding(0)->stream_per_vertex<glm::vec3>()->to_location(0), // inPosition

			cfg::primitive_topology::points,
			cfg::viewport_depth_scissors_config::from_framebuffer(avk::context().main_window()->backbuffer_reference_at_index(0)),
			cfg::depth_write::disabled(),
			//cfg::depth_test::disabled(),

			attachment::declare(format_from_window_color_buffer(context().main_window()), on_load::load, usage::color(0),      on_store::store),
			attachment::declare(format_from_window_depth_buffer(context().main_window()), on_load::load, usage::depth_stencil, on_store::store),

			push_constant_binding_data{ shader_type::vertex | shader_type::fragment, 0, sizeof(PushConstantsVisPath) }
		);

		// create an updater, and add the pipelines (needed for window resize)
		mUpdater.emplace();
		mUpdater->on(avk::swapchain_changed_event(avk::context().main_window())).update(mPipelineVisPath1);

		// create a command pool
		mCommandPool = avk::context().create_command_pool(mQueue->family_index(), vk::CommandPoolCreateFlagBits::eTransient);

		// create vertex buffers
		for (avk::window::frame_id_t fif = 0; fif < avk::context().main_window()->number_of_frames_in_flight(); ++fif) {
			mVertexBufferVisPath1.push_back(avk::context().create_buffer(avk::memory_usage::device, {}, avk::vertex_buffer_meta::create_from_element_size(sizeof(glm::vec3), MAX_POINTS_TO_VISUALIZE)        .describe_only_member(glm::vec3(0), avk::content_description::position)));
			mVertexBufferVisPath2.push_back(avk::context().create_buffer(avk::memory_usage::device, {}, avk::vertex_buffer_meta::create_from_element_size(sizeof(glm::vec3), MAX_CONTROL_POINTS_TO_VISUALIZE).describe_only_member(glm::vec3(0), avk::content_description::position)));
		}

		mRenderingInited = true;
	}

	preset_data * find_preset(std::string aName) {
		for (auto &p : mPresets) {
			if (p.name == aName) return &p;
		}
		return nullptr;
	}

	void invoke_preset(preset_data *aPreset) {
		stop_all_motion();
		auto quakeCam = avk::current_composition()->element_by_type<avk::quake_camera>();
		auto orbitCam = avk::current_composition()->element_by_type<avk::orbit_camera>();
		if (!quakeCam || !orbitCam || !aPreset) return;
		if (aPreset->type == preset_type::location) {
			quakeCam->set_translation(aPreset->translation);
			quakeCam->set_rotation(aPreset->rotation);
			orbitCam->set_translation(aPreset->translation);
			orbitCam->set_rotation(aPreset->rotation);
		} else if (aPreset->type == preset_type::circular || aPreset->type == preset_type::path) {
			aPreset->motion_start_time = static_cast<float>(avk::context().get_time());
			aPreset->motion_active = true;
		}
	}

	std::string float_to_string(float aFloat) {
		static char buf[128];
		snprintf(buf, _ARRAYSIZE(buf), "%.3ff", aFloat);
		return std::string(buf);
	}

	std::string vec2_to_string(glm::vec2 &v) {
		return "{" + float_to_string(v.x) + "," + float_to_string(v.y) + "}";
	}
	std::string vec3_to_string(glm::vec3 &v) {
		return "{" + float_to_string(v.x) + "," + float_to_string(v.y) + "," + float_to_string(v.z) + "}";
	}

	std::string focus_type_to_string(focus_type aFocusType) {
		switch(aFocusType) {
		case focus_type::towards_point:		return "camera_presets::focus_type::towards_point";
		case focus_type::away_from_point:	return "camera_presets::focus_type::away_from_point";
		case focus_type::forward:			return "camera_presets::focus_type::forward";
		case focus_type::backward:			return "camera_presets::focus_type::backward";
		case focus_type::none:				return "camera_presets::focus_type::none";
		default:							return "static_cast<camera_presets::focus_type>(" + std::to_string(static_cast<int>(aFocusType)) + ")";
		}
	}

	std::string generate_code(preset_data *aPreset) {
		if (!aPreset) return "";
		const std::string objName = "";

		std::string result = objName;
		if (aPreset->type == preset_type::location) {
			result += "add_location(\"" + aPreset->name + "\", "
				    + vec3_to_string(aPreset->translation) + ", "
				    + vec3_to_string(aPreset->direction) + ");";
		} else if (aPreset->type == preset_type::circular) {
			result += "add_circular_motion(\"" + aPreset->name + "\", "
				+ vec3_to_string(aPreset->center) + ", "
				+ vec2_to_string(aPreset->radius_xz) + ", "
				+ vec3_to_string(aPreset->focus) + ", "
				+ float_to_string(aPreset->angular_speed) + ", "
				+ float_to_string(aPreset->start_angle) + ", "
				+ focus_type_to_string(aPreset->focus_type) + ");";
		} else if (aPreset->type == preset_type::path) {
			result += "add_path(\"" + aPreset->name + "\", ";
			switch(aPreset->path_type) {
			case path_type::bezier:				result += "camera_presets::path_type::bezier";				break;
			case path_type::quadratic_bspline:	result += "camera_presets::path_type::quadratic_bspline";	break;
			case path_type::cubic_bspline:		result += "camera_presets::path_type::cubic_bspline";		break;
			case path_type::catmull_rom:		result += "camera_presets::path_type::catmull_rom";			break;
			default:							result += "static_cast<camera_presets::path_type>(" + std::to_string(static_cast<int>(aPreset->path_type)) + ")"; break;
			}
			result += ", "
				+ float_to_string(aPreset->path_duration) + ", "
				+ (aPreset->path_cyclic ? "true" : "false") + ", "
				+ "{";
			for (size_t i = 0; i < aPreset->path_control_points.size(); ++i) {
				bool last = (i == aPreset->path_control_points.size() - 1);
				if (i % 5 == 0) result += "\n\t";
				result += vec3_to_string(aPreset->path_control_points[i]);
				if (!last) result += ", ";
			}
			result += "\n}, "
				+ focus_type_to_string(aPreset->focus_type) + ", "
				+ vec3_to_string(aPreset->focus) + ");";
		}
		return result + "\n";
	}

	std::string unique_name(std::string aName, preset_data *aPresetToExclude = nullptr) {
		if (aName.length() == 0) aName = "unnamed";
		if (aName.length() > MAX_NAME_LEN) aName = aName.substr(0, MAX_NAME_LEN);
		auto found = find_preset(aName);
		if (!found || found == aPresetToExclude) return aName;
		int cnt = 1;
		while (true) {
			std::string newName = aName + "_" + std::to_string(cnt);
			if (!find_preset(newName)) return newName;
			cnt++;
		}
	}

	void stop_all_motion(preset_data *aPresetToExclude) {
		for (auto &p : mPresets) {
			if (&p != aPresetToExclude) {
				p.motion_active = false;
			}
		}
	}

	glm::quat camera_rotation_from_direction(const glm::vec3 &aDirection) {
		// code taken from transform::look_along()
		if (glm::dot(aDirection, aDirection) < 1.2e-7 /* ~machine epsilon */) return glm::quat();
		return glm::normalize(glm::quatLookAt(glm::normalize(aDirection), glm::vec3{0.f, 1.f, 0.f}));
	}

	bool mAllowAddPresets    = true;
	bool mShowCameraInfo     = true;
	bool mAllowVisualizePath = true;
	ImVec2 mInitialPosition = { 250.0f, 1.0f };
	ImVec2 mInitialSize     = { 0.0f, 0.0f };

	bool mVisualizePath = false;
	std::string mVisualizePathPresetName = "";
	int  mVisualizePathCurrentPointIndex = -1;

	avk::queue* mQueue;
	avk::command_pool mCommandPool;
	avk::graphics_pipeline mPipelineVisPath1, mPipelineVisPath2;
	std::vector<avk::buffer> mVertexBufferVisPath1, mVertexBufferVisPath2;

	bool mRenderingInited = false;
	bool mGuiEnabled = true;

	std::vector<preset_data> mPresets;

	struct PushConstantsVisPath {
		glm::mat4 mViewProjMatrix;
		glm::vec4 mColor;  // .a = point size
		glm::vec4 mColor2;
		int  mVertexToHighlight;
		float pad1,pad2,pad3;
	};

	static const int MAX_NAME_LEN = 127;
	static const int MAX_POINTS_TO_VISUALIZE = 5000;
	static const int MAX_CONTROL_POINTS_TO_VISUALIZE = 500;
};

