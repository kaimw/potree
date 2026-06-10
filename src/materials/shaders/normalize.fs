#version 300 es
precision mediump float;
precision mediump int;

uniform sampler2D uWeightMap;
uniform sampler2D uDepthMap;

in vec2 vUv;

out vec4 fragColor;

void main() {
	float depth = texture(uDepthMap, vUv).r;
	
	if(depth >= 1.0){
		discard;
	}

	fragColor = vec4(depth, 1.0, 0.0, 1.0);

	vec4 color = texture(uWeightMap, vUv); 
	color = color / color.w;
	
	fragColor = vec4(color.xyz, 1.0); 
	
	gl_FragDepth = depth;


}