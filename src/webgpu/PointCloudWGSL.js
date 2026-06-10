// ── Common WGSL helpers shared across all point cloud shaders ──

const wgslCommon = `
fn isBitSet(value : u32, bitIndex : u32) -> bool {
	return (value & (1u << bitIndex)) != 0u;
}

fn numberOfOnesBefore(value : u32, upToExclusive : u32) -> u32 {
	var count = 0u;
	for (var i = 0u; i < 8u; i = i + 1u) {
		if (i >= upToExclusive) {
			break;
		}
		if (isBitSet(value, i)) {
			count = count + 1u;
		}
	}
	return count;
}

fn getChannel(packed : u32, channel : u32) -> u32 {
	return (packed >> (channel * 8u)) & 255u;
}
`;

const wgslUniforms = `
struct CameraUniforms {
	mvp : mat4x4<f32>,
	screenSize : vec2<f32>,
	pointSize : f32,
	nodeSpacing : f32,
	fov : f32,
	minSize : f32,
	maxSize : f32,
	pointSizeType : f32,
	nodeLevel : f32,
	vnStart : f32,
	octreeSize : f32,
	useVisibilityBuffer : f32,
	padding0 : f32,
	padding1 : f32,
	padding2 : f32,
	padding3 : f32,
};

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var<storage, read> visibleNodes : array<u32>;
`;

const wgslGetLOD = `
fn getLOD(position : vec3<f32>) -> f32 {
	if (camera.useVisibilityBuffer < 0.5) {
		return camera.nodeLevel;
	}

	var offset = vec3<f32>(0.0, 0.0, 0.0);
	var iOffset = u32(camera.vnStart);
	var depth = camera.nodeLevel;

	for (var i = 0u; i <= 30u; i = i + 1u) {
		let nodeSizeAtLevel = camera.octreeSize / exp2(f32(i) + camera.nodeLevel);
		let index3d = floor((position - offset) / nodeSizeAtLevel + vec3<f32>(0.5, 0.5, 0.5));
		let childIndex = u32(clamp(round(4.0 * index3d.x + 2.0 * index3d.y + index3d.z), 0.0, 7.0));
		let packed = visibleNodes[min(iOffset, 8191u)];
		let mask = getChannel(packed, 0u);

		if (isBitSet(mask, childIndex)) {
			let advanceG = getChannel(packed, 1u) * 256u;
			let advanceB = getChannel(packed, 2u);
			let advanceChild = numberOfOnesBefore(mask, childIndex);
			iOffset = iOffset + advanceG + advanceB + advanceChild;
			depth = depth + 1.0;
		} else {
			let lodOffset = f32(getChannel(packed, 3u)) / 10.0 - 10.0;
			return depth + lodOffset;
		}

		offset = offset + vec3<f32>(nodeSizeAtLevel * 0.5) * index3d;
	}

	return depth;
}

fn computePointSize(position : vec3<f32>, clip_w : f32) -> f32 {
	let slope = tan(camera.fov * 0.5);
	let projFactor = 0.5 * camera.screenSize.y / max(slope * clip_w, 0.0001);
	var pointSize = camera.pointSize;
	let lodAttenuation = exp2(max(getLOD(position), 0.0));

	if (camera.pointSizeType > 1.5) {
		pointSize = camera.pointSize * camera.nodeSpacing * 1.7 * projFactor / max(lodAttenuation, 1.0);
	} else if (camera.pointSizeType > 0.5) {
		pointSize = camera.pointSize * camera.nodeSpacing * projFactor;
	}

	return clamp(pointSize, camera.minSize, camera.maxSize);
}
`;

// ── Standard (square) point cloud shader ──

export const pointCloudVertexWGSL = `
${wgslCommon}
${wgslUniforms}
${wgslGetLOD}

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec3<f32>,
	@location(1) pointCoord : vec2<f32>,
};

@vertex fn main(
	@builtin(vertex_index) vertexIndex : u32,
	@builtin(instance_index) instanceIndex : u32,
	@location(0) position : vec3<f32>,
	@location(1) color : vec3<f32>
) -> VertexOutput {
	let corners = array<vec2<f32>, 4>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>( 1.0, -1.0),
		vec2<f32>(-1.0,  1.0),
		vec2<f32>( 1.0,  1.0)
	);

	let corner = corners[vertexIndex];
	let clip = camera.mvp * vec4<f32>(position, 1.0);
	let pixelScale = vec2<f32>(2.0 / camera.screenSize.x, 2.0 / camera.screenSize.y);
	let pointSize = computePointSize(position, clip.w);
	let offset = vec2<f32>(corner.x * pointSize * 0.5, corner.y * pointSize * 0.5) * pixelScale * clip.w;

	var output : VertexOutput;
	output.position = vec4<f32>(clip.xy + offset, clip.z, clip.w);
	output.color = color;
	output.pointCoord = corner * 0.5 + vec2<f32>(0.5, 0.5);
	return output;
}
`;

export const pointCloudFragmentWGSL = `
struct FragmentInput {
	@location(0) color : vec3<f32>,
	@location(1) pointCoord : vec2<f32>,
};

@fragment fn main(input : FragmentInput) -> @location(0) vec4<f32> {
	return vec4<f32>(input.color, 1.0);
}
`;

// ── HQ Depth Pass (circular points, extended depth, output depth to color) ──

export const depthPassVertexWGSL = `
${wgslCommon}
${wgslUniforms}
${wgslGetLOD}

struct DepthVertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec3<f32>,
	@location(1) pointCoord : vec2<f32>,
	@location(2) @interpolate(flat) vRadius : f32,
	@location(3) @interpolate(flat) projFactor : f32,
};

@vertex fn main(
	@builtin(vertex_index) vertexIndex : u32,
	@builtin(instance_index) instanceIndex : u32,
	@location(0) position : vec3<f32>,
	@location(1) color : vec3<f32>
) -> DepthVertexOutput {
	let corners = array<vec2<f32>, 4>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>( 1.0, -1.0),
		vec2<f32>(-1.0,  1.0),
		vec2<f32>( 1.0,  1.0)
	);

	let corner = corners[vertexIndex];
	let clip = camera.mvp * vec4<f32>(position, 1.0);
	let pixelScale = vec2<f32>(2.0 / camera.screenSize.x, 2.0 / camera.screenSize.y);
	let pointSize = computePointSize(position, clip.w);

	let slope = tan(camera.fov * 0.5);
	let pf = 0.5 * camera.screenSize.y / max(slope * clip.w, 0.0001);
	let radius = pointSize / pf;

	let offset = vec2<f32>(corner.x * pointSize * 0.5, corner.y * pointSize * 0.5) * pixelScale * clip.w;

		// Extend depth by 2 * vRadius (matching WebGL view-space extension)
		let extendedZ = clip.z + 2.0 * radius;
		let extendedW = clip.w + 2.0 * radius;
		let extendAdjust = extendedW / clip.w;
		let extendedClip = vec4<f32>((clip.xy + offset) * extendAdjust, extendedZ, extendedW);

	var output : DepthVertexOutput;
	output.position = extendedClip;
	output.color = color;
	output.pointCoord = corner * 0.5 + vec2<f32>(0.5, 0.5);
	output.vRadius = radius;
	output.projFactor = pf;
	return output;
}
`;

export const depthPassFragmentWGSL = `
struct DepthFragmentInput {
	@location(0) color : vec3<f32>,
	@location(1) pointCoord : vec2<f32>,
	@location(2) @interpolate(flat) vRadius : f32,
	@location(3) @interpolate(flat) projFactor : f32,
};

@fragment fn main(input : DepthFragmentInput) -> @location(0) vec4<f32> {
	// Circular point: discard fragments outside the circle
	let dist = length(input.pointCoord - vec2<f32>(0.5, 0.5));
	if (dist > 0.5) {
		discard;
	}

		// Stores pointCoord.x in R channel (unused, kept for EDL compatibility)
	return vec4<f32>(input.pointCoord.x, 1.0, 0.0, 1.0);
}
`;

// ── HQ Attribute Pass (circular points, weighted color + additive blend) ──

export const attributePassVertexWGSL = `
${wgslCommon}
${wgslUniforms}
${wgslGetLOD}

struct AttributeVertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec3<f32>,
	@location(1) pointCoord : vec2<f32>,
};

@vertex fn main(
	@builtin(vertex_index) vertexIndex : u32,
	@builtin(instance_index) instanceIndex : u32,
	@location(0) position : vec3<f32>,
	@location(1) color : vec3<f32>
) -> AttributeVertexOutput {
	let corners = array<vec2<f32>, 4>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>( 1.0, -1.0),
		vec2<f32>(-1.0,  1.0),
		vec2<f32>( 1.0,  1.0)
	);

	let corner = corners[vertexIndex];
	let clip = camera.mvp * vec4<f32>(position, 1.0);
	let pixelScale = vec2<f32>(2.0 / camera.screenSize.x, 2.0 / camera.screenSize.y);
	let pointSize = computePointSize(position, clip.w);
	let offset = vec2<f32>(corner.x * pointSize * 0.5, corner.y * pointSize * 0.5) * pixelScale * clip.w;

	var output : AttributeVertexOutput;
	output.position = vec4<f32>(clip.xy + offset, clip.z, clip.w);
	output.color = color;
	output.pointCoord = corner * 0.5 + vec2<f32>(0.5, 0.5);
	return output;
}
`;

export const attributePassFragmentWGSL = `
struct AttributeFragmentInput {
	@location(0) color : vec3<f32>,
	@location(1) pointCoord : vec2<f32>,
};

@fragment fn main(input : AttributeFragmentInput) -> @location(0) vec4<f32> {
	// Circular point: smooth falloff at edges
	let dist = length(input.pointCoord - vec2<f32>(0.5, 0.5));
	if (dist > 0.5) {
		discard;
	}
		// Parabolic falloff: weight = max(0, 1-2*dist)^1.5 (matching WebGL weighted_splats)
		let d = 2.0 * dist;
		let weight = max(0.0, 1.0 - d);
		let w = pow(weight, 1.5);

	// Weighted splat (matching WebGL): output (color*w, w), blend src_alpha/one
	return vec4<f32>(input.color * w, w);
}
`;

// ── Normalization Pass (fullscreen quad) ──

export const normalizeVertexWGSL = `
struct NormalizeVertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) uv : vec2<f32>,
};

@vertex fn main(
	@builtin(vertex_index) vertexIndex : u32
) -> NormalizeVertexOutput {
	// Fullscreen triangle covering NDC [-1,1]^2. UVs flipped: WebGPU
	// framebuffer origin is top-left, NDC y points up, so invert v.
	let pos = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>( 3.0, -1.0),
		vec2<f32>(-1.0,  3.0)
	);
	let uvs = array<vec2<f32>, 3>(
		vec2<f32>(0.0, 1.0),
		vec2<f32>(2.0, 1.0),
		vec2<f32>(0.0, -1.0)
	);

	var output : NormalizeVertexOutput;
	output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
	output.uv = uvs[vertexIndex];
	return output;
}
`;

export const normalizeFragmentWGSL = `
@group(0) @binding(0) var uWeightMap : texture_2d<f32>;
@group(0) @binding(1) var uDepthMap : texture_2d<f32>;
@group(0) @binding(2) var uSampler : sampler;

struct NormalizeFragmentInput {
	@location(0) uv : vec2<f32>,
};

@fragment fn main(input : NormalizeFragmentInput) -> @location(0) vec4<f32> {
	let weight = textureSample(uWeightMap, uSampler, input.uv);

	// Discard background where no point was rendered
	if (weight.a < 0.0001) {
		discard;
	}

	// Avoid division by zero
	let alpha = max(weight.a, 0.0001);
	let color = weight.rgb / alpha;

	return vec4<f32>(color, 1.0);
}
`;
