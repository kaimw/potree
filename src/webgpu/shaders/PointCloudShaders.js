export const pointCloudVertexWGSL = `
struct CameraUniforms {
	viewProjection : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera : CameraUniforms;

struct VertexInput {
	@location(0) position : vec3<f32>,
	@location(1) color : vec3<f32>,
};

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec3<f32>,
};

@vertex fn main(input : VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = camera.viewProjection * vec4<f32>(input.position, 1.0);
	output.color = input.color;
	return output;
}
`;

export const pointCloudFragmentWGSL = `
struct FragmentInput {
	@location(0) color : vec3<f32>,
};

@fragment fn main(input : FragmentInput) -> @location(0) vec4<f32> {
	return vec4<f32>(input.color, 1.0);
}
`;

export const normalizeFragmentWGSL = `
@fragment fn normalizeMain(
	@location(0) weightedColor : vec4<f32>,
	@location(1) weight : f32
) -> @location(0) vec4<f32> {
	if (weight <= 0.0) {
		return vec4<f32>(0.0, 0.0, 0.0, 1.0);
	}
	return vec4<f32>(weightedColor.rgb / weight, 1.0);
}
`;
